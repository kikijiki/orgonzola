import React from "react"
import ReactDOM from "react-dom/client"
import App from "./App"
import { ConfirmProvider } from "./components/Confirm"
import { ToastProvider } from "./components/Toast"
import "./index.css"

const root = document.getElementById("root")
if (!root) throw new Error("missing #root element")

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <ToastProvider>
      <ConfirmProvider>
        <App />
      </ConfirmProvider>
    </ToastProvider>
  </React.StrictMode>,
)
