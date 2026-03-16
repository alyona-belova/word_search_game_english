#!/usr/bin/env node
// ---------------------------------------------------------------------------
// Usage:
//   METRICA_TOKEN=<oauth_token> METRICA_COUNTER=<counter_id> node fetch-logs.js
//
// How to get an OAuth token:
//   https://yandex.ru/dev/metrika/ru/management/access
//   Register an app with "метрика" scope, then get a token via OAuth flow.
// ---------------------------------------------------------------------------

const https = require("https");

const TOKEN = process.env.METRICA_TOKEN;
const COUNTER = process.env.METRICA_COUNTER;

if (!TOKEN || !COUNTER) {
  console.error("Set METRICA_TOKEN and METRICA_COUNTER environment variables.");
  process.exit(1);
}

// Date range — last 30 days by default
const today = new Date();
const thirtyAgo = new Date(today - 30 * 24 * 60 * 60 * 1000);
const fmt = (d) => d.toISOString().slice(0, 10);
const DATE1 = fmt(thirtyAgo);
const DATE2 = fmt(today);

//  time on site → ym:s:visitDuration
//  revisit rate → ym:s:isNewUser
//  hint_received goal → ym:s:goalReaches
//  drop_off goal → ym:s:goalReaches
//  level_started/completed → ym:s:goalReaches
//  custom params → ym:s:params
const FIELDS = [
  "ym:s:visitID",
  "ym:s:date",
  "ym:s:clientID",
  "ym:s:isNewUser",
  "ym:s:visitDuration",
  "ym:s:goalReaches",
  "ym:s:params",
].join(",");

function apiRequest(method, path, body) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: "api-metrika.yandex.net",
      path,
      method,
      headers: {
        Authorization: `OAuth ${TOKEN}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
    };
    const req = https.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode, body: data });
        }
      });
    });
    req.on("error", reject);
    if (body) req.write(body);
    req.end();
  });
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function main() {
  console.log(`Requesting logs for counter ${COUNTER}, ${DATE1} → ${DATE2}`);
  const createRes = await apiRequest(
    "POST",
    `/logs/v1/counter/${COUNTER}/logrequest?` +
      new URLSearchParams({
        date1: DATE1,
        date2: DATE2,
        source: "visits",
        fields: FIELDS,
      }),
  );

  if (createRes.status !== 200) {
    console.error("Failed to create log request:", createRes.body);
    process.exit(1);
  }

  const requestId = createRes.body.log_request?.request_id;
  if (!requestId) {
    console.error("No request_id in response:", createRes.body);
    process.exit(1);
  }
  console.log("Log request ID:", requestId);

  let status = "";
  while (status !== "processed") {
    await sleep(10_000);
    const infoRes = await apiRequest(
      "GET",
      `/logs/v1/counter/${COUNTER}/logrequest/${requestId}`,
    );
    status = infoRes.body.log_request?.status ?? "";
    console.log("Status:", status);
    if (status === "cleaned_by_user" || status === "processing_failed") {
      console.error("Log request failed:", infoRes.body);
      process.exit(1);
    }
  }
  const partsRes = await apiRequest(
    "GET",
    `/logs/v1/counter/${COUNTER}/logrequest/${requestId}`,
  );
  const parts = partsRes.body.log_request?.parts ?? [];
  console.log(`Downloading ${parts.length} part(s)…`);

  const rows = [];
  for (const part of parts) {
    const partRes = await apiRequest(
      "GET",
      `/logs/v1/counter/${COUNTER}/logrequest/${requestId}/part/${part.part_number}/download`,
    );
    const lines = (
      typeof partRes.body === "string"
        ? partRes.body
        : JSON.stringify(partRes.body)
    )
      .split("\n")
      .filter(Boolean);
    if (rows.length === 0) {
      rows.push(lines[0]);
    }
    rows.push(...lines.slice(1));
  }

  const fs = require("fs");
  const outFile = `metrica-logs-${DATE1}-${DATE2}.tsv`;
  fs.writeFileSync(outFile, rows.join("\n"));
  console.log(`Saved ${rows.length - 1} sessions to ${outFile}`);

  await apiRequest(
    "POST",
    `/logs/v1/counter/${COUNTER}/logrequest/${requestId}/clean`,
  );
  console.log("Log request cleaned.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
