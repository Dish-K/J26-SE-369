import Link from "next/link";

const steps = [
  {
    number: "01",
    title: "Observe the work",
    description:
      "Capture how a candidate writes and revises code during a live interview, with consent built into the process.",
  },
  {
    number: "02",
    title: "Surface useful context",
    description:
      "Study behavioral patterns and make the supporting evidence understandable to the interviewer.",
  },
  {
    number: "03",
    title: "Continue the conversation",
    description:
      "Use focused follow-up questions to explore a candidate’s understanding of their own solution.",
  },
];

const researchAreas = [
  {
    label: "Telemetry infrastructure",
    title: "The shape of the work.",
    description:
      "Collect coding activity and turn it into structured behavioral features for analysis.",
  },
  {
    label: "Behavioral classifier",
    title: "Patterns worth reviewing.",
    description:
      "Analyze agreed signals and identify moments that may need a closer look.",
  },
  {
    label: "Adaptive probing",
    title: "A better next question.",
    description:
      "Generate targeted follow-ups after submission to test understanding in context.",
  },
  {
    label: "Explainability",
    title: "Evidence people can use.",
    description:
      "Present the reasons behind a signal so an interviewer can make an informed judgment.",
  },
];

export default function Home() {
  return (
    <>
      <header className="site-header page-gutter">
        <Link className="wordmark" href="/" aria-label="CodeTrace home">
          CodeTrace<span className="wordmark-period">.</span>
        </Link>
        <nav className="site-nav" aria-label="Main navigation">
          <a href="#approach">Approach</a>
          <a href="#research">Research areas</a>
        </nav>
        <Link className="header-login" href="/login">
          Log in <span aria-hidden="true">↗</span>
        </Link>
      </header>

      <main>
        <section className="hero page-gutter" aria-labelledby="hero-title">
          <div className="hero-kicker">
            <span className="eyebrow">CodeTrace / Research platform</span>
            <span className="eyebrow hero-kicker-right">Integrity for technical interviews</span>
          </div>
          <div className="hero-grid">
            <div className="hero-copy">
              <h1 id="hero-title">
                See the <em>process</em> behind the code.
              </h1>
              <p>
                CodeTrace explores how coding behavior can help interviewers understand a candidate’s work in real time, and ask more meaningful questions.
              </p>
              <div className="hero-actions">
                <Link className="button button-accent" href="/login">
                  Go to login <span aria-hidden="true">↗</span>
                </Link>
                <a className="text-link" href="#approach">
                  Explore the approach <span aria-hidden="true">↓</span>
                </a>
              </div>
            </div>

            <div className="concept-frame" aria-label="Concept illustration of a CodeTrace interview review">
              <div className="concept-toolbar">
                <span>CODETRACE / REVIEW VIEW</span>
                <span className="concept-status"><i aria-hidden="true" /> CONCEPT</span>
              </div>
              <div className="concept-body">
                <div className="concept-title-row">
                  <div>
                    <span className="concept-overline">INTERVIEW ACTIVITY</span>
                    <strong>Solution in progress</strong>
                  </div>
                  <span className="concept-time">00:18:42</span>
                </div>
                <div className="concept-timeline" aria-hidden="true">
                  <span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span />
                </div>
                <div className="concept-code">
                  <div className="concept-code-header">
                    <span>solution.py</span>
                    <span>PYTHON</span>
                  </div>
                  <div className="concept-code-lines" aria-hidden="true">
                    <span>01</span><code><b>def</b> find_pair(numbers, target):</code>
                    <span>02</span><code>    seen = {'{}'}</code>
                    <span>03</span><code>    <b>for</b> index, value <b>in</b> enumerate(numbers):</code>
                    <span>04</span><code>        match = target - value</code>
                    <span>05</span><code>        <b>if</b> match <b>in</b> seen:</code>
                    <span>06</span><code>            <b>return</b> [seen[match], index]</code>
                    <span>07</span><code>        seen[value] = index</code>
                  </div>
                </div>
                <div className="concept-note">
                  <span className="concept-note-mark" aria-hidden="true">↗</span>
                  <div>
                    <strong>Ask about the approach</strong>
                    <p>Follow up on the reasoning behind a change.</p>
                  </div>
                </div>
              </div>
              <div className="concept-foot">A visual concept of the planned interviewer experience</div>
            </div>
          </div>
          <div className="hero-bottom">
            <span>BUILT FOR HUMAN REVIEW</span>
            <span>RESEARCH PROJECT · J26-SE-369</span>
          </div>
        </section>

        <section className="manifesto page-gutter" aria-label="CodeTrace principle">
          <p>Code can show <strong>what</strong> happened. The interview should help reveal <strong>how</strong> and <strong>why</strong>.</p>
        </section>

        <section className="approach section-shell page-gutter" id="approach" aria-labelledby="approach-title">
          <div className="section-intro">
            <span className="eyebrow">The approach</span>
            <h2 id="approach-title">More context.<br />Better questions.</h2>
            <p>CodeTrace is being designed to support the interviewer’s judgment throughout a live technical interview.</p>
          </div>
          <div className="step-list">
            {steps.map((step) => (
              <article className="step" key={step.number}>
                <span className="step-number">{step.number}</span>
                <div>
                  <h3>{step.title}</h3>
                  <p>{step.description}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="research section-shell page-gutter" id="research" aria-labelledby="research-title">
          <div className="research-heading">
            <div>
              <span className="eyebrow">The research</span>
              <h2 id="research-title">Four connected areas.<br />One clearer picture.</h2>
            </div>
            <p>Each part of CodeTrace contributes a different piece of context, from activity capture to explanations an interviewer can use.</p>
          </div>
          <div className="research-grid">
            {researchAreas.map((area) => (
              <article className="research-card" key={area.label}>
                <span className="eyebrow">{area.label}</span>
                <h3>{area.title}</h3>
                <p>{area.description}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="principle page-gutter" aria-labelledby="principle-title">
          <span className="eyebrow">A guiding principle</span>
          <h2 id="principle-title">A signal starts a conversation.<br />It does not make the decision.</h2>
          <p>CodeTrace is intended to give interviewers evidence and follow-up prompts, while keeping the final judgment with people.</p>
        </section>

        <section className="closing page-gutter" aria-labelledby="closing-title">
          <div>
            <span className="eyebrow">CodeTrace</span>
            <h2 id="closing-title">A closer look at how people code.</h2>
          </div>
          <Link className="button button-accent" href="/login">
            Go to login <span aria-hidden="true">↗</span>
          </Link>
        </section>
      </main>

      <footer className="site-footer page-gutter">
        <Link className="wordmark" href="/" aria-label="CodeTrace home">
          CodeTrace<span className="wordmark-period">.</span>
        </Link>
        <span>Integrity for technical interviews</span>
        <span>Research project · J26-SE-369</span>
      </footer>
    </>
  );
}
