import { useEffect, useState } from 'react';

export type Page = 'models' | 'predict' | 'insights' | 'audit';
const PAGES: Page[] = ['models', 'predict', 'insights', 'audit'];

const parse = (): Page => {
  const name = window.location.hash.replace(/^#\/?/, '') as Page;
  return PAGES.includes(name) ? name : 'models';
};

/** Minimal hash router (#/models, #/predict, ...): pages survive reloads without a router dependency. */
export const useHashRoute = (): [Page, (page: Page) => void] => {
  const [page, setPage] = useState<Page>(parse);

  useEffect(() => {
    const onChange = () => setPage(parse());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  const navigate = (next: Page) => {
    window.location.hash = `/${next}`;
  };
  return [page, navigate];
};
