declare module 'langdetect' {
  export function detect(text: string): Array<{ lang: string; prob: number }>;
}