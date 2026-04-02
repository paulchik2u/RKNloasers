export default {
  async fetch(request, env, ctx) {
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "x-phantom-version, content-type",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    if (request.method !== "POST") {
      return new Response("Method not allowed", {
        status: 403,
        headers: { "Content-Type": "text/plain" },
      });
    }

    const version = request.headers.get("x-phantom-version");
    if (!version) {
      return new Response("Missing x-phantom-version header", {
        status: 400,
        headers: { "Content-Type": "text/plain" },
      });
    }

    const exitNodeUrl = env.EXIT_NODE_URL;
    if (!exitNodeUrl) {
      return new Response("Exit node not configured", {
        status: 500,
        headers: { "Content-Type": "text/plain" },
      });
    }

    try {
      const body = await request.arrayBuffer();

      const exitResponse = await fetch(exitNodeUrl, {
        method: "POST",
        headers: {
          "x-phantom-version": version,
          "content-type": "application/octet-stream",
        },
        body: body,
      });

      const responseHeaders = new Headers();
      responseHeaders.set("Access-Control-Allow-Origin", "*");
      responseHeaders.set("Content-Type", "application/octet-stream");

      if (exitResponse.headers.has("x-phantom-status")) {
        responseHeaders.set("x-phantom-status", exitResponse.headers.get("x-phantom-status"));
      }

      return new Response(exitResponse.body, {
        status: exitResponse.status,
        headers: responseHeaders,
      });
    } catch (err) {
      return new Response("Exit node unreachable", {
        status: 502,
        headers: {
          "Content-Type": "text/plain",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }
  },
};
