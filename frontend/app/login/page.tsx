import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Login",
  description: "CodeTrace login page.",
};

export default function LoginPage() {
  return (
    <div className="placeholder-page">
      <header className="placeholder-header page-gutter">
        <Link className="wordmark" href="/" aria-label="CodeTrace home">
          CodeTrace<span className="wordmark-period">.</span>
        </Link>
        <Link className="text-link" href="/">
          Back to home <span aria-hidden="true">↗</span>
        </Link>
      </header>
      <main className="placeholder-main page-gutter">
        <span className="eyebrow">CodeTrace / Account</span>
        <h1>Login</h1>
      </main>
    </div>
  );
}
