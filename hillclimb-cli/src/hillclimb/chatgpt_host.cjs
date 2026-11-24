// This executable stays in the trusted desktop process. The DevKit owns OAuth.
const { join, resolve } = require("node:path");
const { pathToFileURL } = require("node:url");

const INPUT_BYTES_MAX = 1024 * 1024;

async function run(client, action, args, input, output) {
  switch (action) {
    case "login": {
      const session = await client.signIn({ reconsent: args[0] === "--reconsent" });
      writeSession(output, session);
      return session.sharing ? 0 : 2;
    }
    case "status":
      writeSession(output, await client.getSession());
      return 0;
    case "logout":
      await client.disconnect();
      writeSession(output, await client.getSession());
      return 0;
    case "models":
      await requirePlan(client);
      output.write(JSON.stringify(await client.listModels(), null, 2) + "\n");
      return 0;
    case "respond": {
      await requirePlan(client);
      const [model, ...flags] = args;
      if (!model) throw new Error("a model slug is required");
      const chunks = [];
      let bytes = 0;
      for await (const chunk of input) {
        bytes += chunk.length;
        if (bytes > INPUT_BYTES_MAX) throw new Error("prompt exceeds 1 MiB");
        chunks.push(chunk);
      }
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw.trim()) throw new Error("pass a prompt on stdin");
      const { input: prompt, instructions } = flags.includes("--json") ? parsePrompt(raw) : { input: raw };
      const stream = flags.includes("--stream");
      const { text } = await client.streamResponse({
        model,
        input: prompt,
        instructions,
        onDelta: stream ? (delta) => output.write(delta) : undefined,
      });
      output.write(stream ? "\n" : text + "\n");
      return 0;
    }
    default:
      throw new Error("unknown ChatGPT command");
  }
}

function parsePrompt(raw) {
  let payload;
  try {
    payload = JSON.parse(raw);
  } catch {
    throw new Error("expected a JSON prompt");
  }
  if (!payload || Array.isArray(payload) || typeof payload !== "object" ||
      Object.keys(payload).some((key) => key !== "input" && key !== "instructions") ||
      !(typeof payload.input === "string" && payload.input.trim() ||
        Array.isArray(payload.input) && payload.input.length && payload.input.every(validMessage)) ||
      (payload.instructions !== undefined && typeof payload.instructions !== "string")) {
    throw new Error("expected text or a nonempty list of text messages in input");
  }
  return payload;
}

function validMessage(message) {
  return message && typeof message === "object" && Object.keys(message).length === 2 &&
    ["user", "assistant", "developer"].includes(message.role) && typeof message.content === "string";
}

function writeSession(output, session) {
  output.write(JSON.stringify({
    status: session.status,
    chatgpt_plan: session.sharing,
    ...(session.error?.code ? { error: session.error.code } : {}),
    ...(session.profileLabel ? { profile: session.profileLabel } : {}),
  }, null, 2) + "\n");
}

async function requirePlan(client) {
  const session = await client.getSession();
  if (!session.sharing) {
    throw new Error("connect an eligible ChatGPT plan with hillclimb inference chatgpt login");
  }
}

async function main(app, safeStorage, shell) {
  const [devkit, action, ...args] = process.argv.slice(2);
  app.setName("Hillclimb");
  await app.whenReady();
  const encryption = safeStorage;
  if (process.platform !== "darwin" || !encryption.isEncryptionAvailable()) {
    throw new Error("the ChatGPT host requires macOS with Keychain available");
  }
  const entry = pathToFileURL(resolve(devkit, "packages/local/dist/index.js"));
  const { createChatGPT } = await import(entry.href);
  const client = createChatGPT({
    appName: "Hillclimb",
    appId: "hillclimb",
    redirectPort: 0,
    sendHostId: true,
    storageDir: join(app.getPath("appData"), "Hillclimb", "chatgpt"),
    openBrowser: (url) => shell.openExternal(url),
    credentialEncryption: {
      id: "hillclimb-electron-safestorage-v1",
      isAvailable: () => encryption.isEncryptionAvailable(),
      encrypt: (text) => encryption.encryptString(text),
      decrypt: (ciphertext) => encryption.decryptString(Buffer.from(ciphertext)),
    },
  });
  return run(client, action, args, process.stdin, process.stdout);
}

if (process.type === "browser" && process.versions.electron) {
  const { app, safeStorage, shell } = require("electron");
  main(app, safeStorage, shell).then((code) => {
    // Settle stdout writes before exiting the desktop host.
    process.stdout.write("", () => app.exit(code));
  }, (error) => {
    // SDK errors are sanitized by the official DevKit. Avoid stack traces and raw payloads.
    const message = (typeof error?.toJSON === "function" || error?.constructor === Error)
      ? error.message : "could not run the ChatGPT host";
    process.stderr.write(`ChatGPT: ${message}\n`, () => app.exit(1));
  });
}
module.exports = { run };
