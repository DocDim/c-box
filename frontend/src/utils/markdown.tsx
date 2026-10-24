import React from 'react';

const bold = (s: string) =>
  s.split(/(\*\*[^*]+\*\*)/g).map((part, j) =>
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={j} className="text-white">{part.slice(2, -2)}</strong>
      : <React.Fragment key={j}>{part}</React.Fragment>
  );

/** Minimal markdown: ## headings, bullet lists and **bold** (LLM output is plain text otherwise). */
export const renderMarkdown = (text: string) =>
  text.split('\n').map((line, i) => {
    const heading = line.match(/^#{1,4}\s+(.*)/);
    if (heading) return <h4 key={i} className="text-indigo-300 font-semibold mt-3 mb-1">{bold(heading[1])}</h4>;
    const bullet = line.match(/^\s*[-*•]\s+(.*)/);
    if (bullet) return <li key={i} className="ml-4 list-disc">{bold(bullet[1])}</li>;
    if (!line.trim()) return <div key={i} className="h-2" />;
    return <p key={i}>{bold(line)}</p>;
  });
