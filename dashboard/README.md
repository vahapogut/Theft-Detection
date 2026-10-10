# TheftGuard Dashboard

Dashboard built with Next.js 16 and React 19. See the [main README](../README.md) for installation, account creation, and server configuration.

```bash
npm ci
npm run dev
```

Browser requests are forwarded to the FastAPI server through `/api` by default. `BACKEND_URL` sets the backend destination; the optional `NEXT_PUBLIC_API_URL` supports a separately deployed API. A session cookie controls access to the API, WebSocket streams, and evidence. Create the first administrator account from the server terminal.

Pages include live video and statistics, camera health, zones and privacy masks, face lists, event review, system settings, and user management for administrators. Fonts are local; builds do not download external fonts.

```bash
npm run lint
npx tsc --noEmit
npm run build
npm run start
```

## Contributors

- [vahapogut](https://github.com/vahapogut)
