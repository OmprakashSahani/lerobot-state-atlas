import {
  MAX_ENVIRONMENT_SPLATS,
  MAX_SPZ_HEADER_OUTPUT_BYTES,
  SPZ_ALLOWED_FLAGS,
  SPZ_FRACTIONAL_BITS_MAX,
  SPZ_FRACTIONAL_BITS_MIN,
  SPZ_HEADER_BYTES,
  SPZ_MAGIC,
  SPZ_SH_DEGREE,
  SPZ_VERSION,
} from "./limits";

export class SpzPreflightError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpzPreflightError";
  }
}

export interface ValidatedSpzHeader {
  version: 3;
  splatCount: number;
  shDegree: 0;
  fractionalBits: number;
  flags: number;
}

export async function decompressSpzHeader(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") {
    throw new SpzPreflightError("This browser cannot inspect gzip-compressed SPZ data.");
  }
  let compressedOffset = 0;
  const source = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (compressedOffset >= bytes.byteLength) {
        controller.close();
        return;
      }
      const end = Math.min(compressedOffset + 64, bytes.byteLength);
      controller.enqueue(bytes.subarray(compressedOffset, end));
      compressedOffset = end;
    },
  }, { highWaterMark: 0 });
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  try {
    const decompressor = new DecompressionStream("gzip") as unknown as TransformStream<
      Uint8Array,
      Uint8Array
    >;
    reader = source.pipeThrough(decompressor).getReader();
    const header = new Uint8Array(SPZ_HEADER_BYTES);
    let copied = 0;
    let produced = 0;
    while (copied < SPZ_HEADER_BYTES) {
      const { done, value } = await reader.read();
      if (done) break;
      produced += value.byteLength;
      if (produced > MAX_SPZ_HEADER_OUTPUT_BYTES) {
        throw new SpzPreflightError("SPZ header inspection exceeded its decompression cap.");
      }
      const take = Math.min(value.byteLength, SPZ_HEADER_BYTES - copied);
      header.set(value.subarray(0, take), copied);
      copied += take;
    }
    if (copied !== SPZ_HEADER_BYTES) throw new SpzPreflightError("SPZ header is truncated.");
    return header;
  } catch (error) {
    if (error instanceof SpzPreflightError) throw error;
    throw new SpzPreflightError("SPZ gzip stream is malformed.");
  } finally {
    await reader?.cancel().catch(() => undefined);
  }
}

export async function inspectSpzHeader(bytes: Uint8Array, declaredCount: number): Promise<ValidatedSpzHeader> {
  const header = await decompressSpzHeader(bytes);
  const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
  if (view.getUint32(0, true) !== SPZ_MAGIC) throw new SpzPreflightError("SPZ magic is invalid.");
  const version = view.getUint32(4, true);
  if (version !== SPZ_VERSION) throw new SpzPreflightError("Only SPZ version 3 is allowed by this spike.");
  const splatCount = view.getUint32(8, true);
  if (splatCount < 1 || splatCount > MAX_ENVIRONMENT_SPLATS) throw new SpzPreflightError("SPZ splat count exceeds spike policy.");
  if (splatCount !== declaredCount) throw new SpzPreflightError("SPZ splat count does not match its manifest.");
  const shDegree = view.getUint8(12);
  if (shDegree !== SPZ_SH_DEGREE) throw new SpzPreflightError("Only spherical-harmonic degree 0 is allowed.");
  const fractionalBits = view.getUint8(13);
  if (fractionalBits < SPZ_FRACTIONAL_BITS_MIN || fractionalBits > SPZ_FRACTIONAL_BITS_MAX) {
    throw new SpzPreflightError("SPZ fractionalBits is outside the conservative range 8 through 16.");
  }
  const flags = view.getUint8(14);
  if ((flags & ~SPZ_ALLOWED_FLAGS) !== 0) throw new SpzPreflightError("SPZ contains unsupported flags.");
  if (view.getUint8(15) !== 0) throw new SpzPreflightError("SPZ reserved header byte must be zero.");
  return { version: 3, splatCount, shDegree: 0, fractionalBits, flags };
}
