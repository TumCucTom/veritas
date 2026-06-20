# Veritas — federated fraud intelligence (FLock.io × UK Sovereign AI)
- core/ Python FL service · web/ Next.js UI · contract/ shared API · mock/ standalone server
## Run
1. mock: `cd mock && npm i && npm start`            (:8001)
2. core: `cd core && pip install -e . && uvicorn server.app:app` (:8000)
3. web:  `cd web && npm i && npm run dev`            (:3000)
Point web at mock or core via NEXT_PUBLIC_API_BASE.
