import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { Chunk, File, Change } from "parse-diff";
import minimatch from "minimatch";

const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL");

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  return response.data;
}

async function getFileContent(
  owner: string,
  repo: string,
  path: string,
  ref: string
): Promise<string | null> {
  try {
    const response = await octokit.repos.getContent({
      owner,
      repo,
      path,
      ref,
    });

    if (
      "content" in response.data &&
      typeof response.data.content === "string"
    ) {
      return Buffer.from(response.data.content, "base64").toString();
    }
    return null;
  } catch (error) {
    console.error(`Error fetching file content: ${error}`);
    return null;
  }
}

// Helper function to get line number from a Change
function getLineNumber(change: Change): number | undefined {
  if ("normal" in change) {
    return change.ln2; // Normal changes have ln2
  } else if ("add" in change) {
    return change.ln; // Add changes have ln
  } else if ("del" in change) {
    return change.ln; // Del changes have ln
  }
  return undefined;
}

// Helper function to check if change is an addition or deletion
function isAddOrDel(change: Change): boolean {
  return "add" in change || "del" in change;
}

function createPrompt(
  file: File,
  chunk: Chunk,
  prDetails: PRDetails,
  fullFileContent?: string
): string {
  const contextSection = fullFileContent
    ? `\nFull file context:\n\`\`\`\n${fullFileContent}\n\`\`\`\n`
    : "";

  // Get the line numbers that were actually changed
  const changedLines = new Set(
    chunk.changes
      .filter(isAddOrDel)
      .map(getLineNumber)
      .filter((ln): ln is number => ln !== undefined)
  );

  return `You are an expert code reviewer focused on identifying only critical issues. Instructions:

- Provide the response in following JSON format: {"reviews": [{"lineNumber": <line_number>, "reviewComment": "<review comment>"}]}
- ONLY comment on the most critical issues that fall into these categories:
  1. High-impact bugs that could cause system failures or data corruption
  2. Critical security vulnerabilities that could lead to exploits
  3. Severe performance issues that could cause system bottlenecks
  4. Major architectural flaws that significantly impact maintainability
  5. Critical business logic flaws that could lead to system misbehavior
- You may only comment on the following changed line numbers: ${Array.from(
    changedLines
  ).join(", ")}
- Completely ignore:
  * Style issues or formatting
  * Documentation
  * Minor optimizations
  * Naming conventions
  * Code organization suggestions
  * Any issue that isn't immediately critical
- Only provide comments when you are highly confident (90%+) that the issue is severe
- Write comments in GitHub Markdown format
- Be direct and specific about the severe impact of each issue
- If you don't find any critical issues, return an empty reviews array

Review the following code diff in the file "${
    file.to
  }" considering the pull request context:
  
Pull request title: ${prDetails.title}
Pull request description:

---
${prDetails.description}
---
${contextSection}
Git diff to review:

\`\`\`diff
${chunk.content}
${chunk.changes.map((c) => `${getLineNumber(c) || ""} ${c.content}`).join("\n")}
\`\`\`
`;
}

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string;
  reviewComment: string;
}> | null> {
  const queryConfig = {
    model: "gpt-4o-2024-11-20",
    temperature: 0.1,
    max_tokens: 1000,
    top_p: 1,
    frequency_penalty: 0.1,
    presence_penalty: 0.1,
  };

  try {
    const response = await openai.chat.completions.create({
      ...queryConfig,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content:
            "You are an expert code reviewer. Only respond with the requested JSON format.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
    });

    const res = response.choices[0].message?.content?.trim() || "{}";
    return JSON.parse(res).reviews;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

function createComment(
  file: File,
  chunk: Chunk,
  aiResponses: Array<{
    lineNumber: string;
    reviewComment: string;
  }>
): Array<{ body: string; path: string; line: number }> {
  // Get the valid line numbers from the chunk
  const validLines = new Set(
    chunk.changes
      .filter(isAddOrDel)
      .map(getLineNumber)
      .filter((ln): ln is number => ln !== undefined)
  );

  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }
    const lineNumber = Number(aiResponse.lineNumber);
    // Only create comments for lines that were actually changed
    if (!validLines.has(lineNumber)) {
      return [];
    }
    return {
      body: aiResponse.reviewComment,
      path: file.to,
      line: lineNumber,
    };
  });
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
): Promise<void> {
  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments,
    event: "COMMENT",
  });
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];

  for (const file of parsedDiff) {
    if (file.to === "/dev/null") continue; // Ignore deleted files

    const fullFileContent = await getFileContent(
      prDetails.owner,
      prDetails.repo,
      file.to!,
      `pull/${prDetails.pull_number}/head`
    );

    for (const chunk of file.chunks) {
      const prompt = createPrompt(
        file,
        chunk,
        prDetails,
        fullFileContent || undefined
      );
      const aiResponse = await getAIResponse(prompt);
      if (aiResponse) {
        const newComments = createComment(file, chunk, aiResponse);
        if (newComments) {
          comments.push(...newComments);
        }
      }
    }
  }
  return comments;
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  if (eventData.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  } else if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
  } else {
    console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("No diff found");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  const comments = await analyzeCode(filteredDiff, prDetails);
  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
