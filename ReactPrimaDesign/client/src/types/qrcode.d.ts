declare module 'qrcode' {
  interface QRCodeToStringOptions {
    type?: string;
    color?: {
      dark?: string;
      light?: string;
    };
    errorCorrectionLevel?: 'L' | 'M' | 'Q' | 'H';
    width?: number;
    margin?: number;
    scale?: number;
  }
  
  interface QRCodeToDataURLOptions extends QRCodeToStringOptions {
    rendererOpts?: {
      quality?: number;
    };
  }
  
  interface QRCodeToFileOptions extends QRCodeToDataURLOptions {}
  
  interface QRCodeToBufferOptions extends QRCodeToStringOptions {
    rendererOpts?: {
      quality?: number;
    };
  }
  
  interface QRCodeToCanvasOptions extends QRCodeToStringOptions {}
  
  function toCanvas(
    canvas: HTMLCanvasElement,
    text: string,
    options?: QRCodeToCanvasOptions
  ): Promise<HTMLCanvasElement>;
  
  function toCanvas(
    canvas: HTMLCanvasElement,
    text: string,
    callback: (error: Error | null, canvas: HTMLCanvasElement) => void
  ): void;
  
  function toCanvas(
    canvas: HTMLCanvasElement,
    text: string,
    options: QRCodeToCanvasOptions,
    callback: (error: Error | null, canvas: HTMLCanvasElement) => void
  ): void;
  
  function toCanvas(
    text: string,
    options?: QRCodeToCanvasOptions
  ): Promise<HTMLCanvasElement>;
  
  function toCanvas(
    text: string,
    callback: (error: Error | null, canvas: HTMLCanvasElement) => void
  ): void;
  
  function toCanvas(
    text: string,
    options: QRCodeToCanvasOptions,
    callback: (error: Error | null, canvas: HTMLCanvasElement) => void
  ): void;
  
  function toDataURL(
    text: string,
    options?: QRCodeToDataURLOptions
  ): Promise<string>;
  
  function toDataURL(
    text: string,
    callback: (error: Error | null, url: string) => void
  ): void;
  
  function toDataURL(
    text: string,
    options: QRCodeToDataURLOptions,
    callback: (error: Error | null, url: string) => void
  ): void;
  
  function toString(
    text: string,
    options?: QRCodeToStringOptions
  ): Promise<string>;
  
  function toString(
    text: string,
    callback: (error: Error | null, string: string) => void
  ): void;
  
  function toString(
    text: string,
    options: QRCodeToStringOptions,
    callback: (error: Error | null, string: string) => void
  ): void;
  
  function toFile(
    path: string,
    text: string,
    options?: QRCodeToFileOptions
  ): Promise<void>;
  
  function toFile(
    path: string,
    text: string,
    callback: (error: Error | null) => void
  ): void;
  
  function toFile(
    path: string,
    text: string,
    options: QRCodeToFileOptions,
    callback: (error: Error | null) => void
  ): void;
  
  function toBuffer(
    text: string,
    options?: QRCodeToBufferOptions
  ): Promise<Buffer>;
  
  function toBuffer(
    text: string,
    callback: (error: Error | null, buffer: Buffer) => void
  ): void;
  
  function toBuffer(
    text: string,
    options: QRCodeToBufferOptions,
    callback: (error: Error | null, buffer: Buffer) => void
  ): void;
  
  export {
    toCanvas,
    toDataURL,
    toString,
    toFile,
    toBuffer,
    QRCodeToStringOptions,
    QRCodeToDataURLOptions,
    QRCodeToFileOptions,
    QRCodeToBufferOptions,
    QRCodeToCanvasOptions
  };
}