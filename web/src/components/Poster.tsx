/** A text-free risograph composition for the empty Compare page: overprinted inks, one
 * halftone patch. Red and blue are the two pipelines' colours. */
export function Poster() {
  return (
    <svg className="poster" viewBox="0 0 1200 300" role="presentation" aria-hidden>
      <defs>
        <pattern id="halftone" width="14" height="14" patternUnits="userSpaceOnUse">
          <circle cx="7" cy="7" r="3.2" fill="#00838a" />
        </pattern>
      </defs>
      <g className="inks">
        <rect x="40" y="70" width="300" height="190" fill="url(#halftone)" />
        <circle cx="330" cy="150" r="118" fill="#e8414f" />
        <circle cx="478" cy="150" r="118" fill="#0078bf" />
        <rect x="560" y="44" width="250" height="196" fill="#ffe800" />
        <path d="M 820 260 A 130 130 0 0 1 1080 260 Z" fill="#ff48b0" />
        <rect x="900" y="40" width="70" height="70" fill="#00838a" />
        <circle cx="1110" cy="88" r="44" fill="#ff6c2f" />
      </g>
      <rect x="40" y="276" width="1120" height="6" fill="#1d1b19" />
    </svg>
  );
}
