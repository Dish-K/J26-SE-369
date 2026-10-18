import { KeyClass } from "./codes";

export type KeyClassCode = (typeof KeyClass)[keyof typeof KeyClass];

const GROUPS: [KeyClassCode, string[]][] = [
  [
    KeyClass.symbol,
    [
      "Minus", "Equal", "BracketLeft", "BracketRight", "Backslash", "Semicolon", "Quote",
      "Backquote", "Comma", "Period", "Slash", "IntlBackslash",
      "NumpadAdd", "NumpadSubtract", "NumpadMultiply", "NumpadDivide", "NumpadDecimal",
    ],
  ],
  [KeyClass.space, ["Space"]],
  [KeyClass.enter, ["Enter", "NumpadEnter"]],
  [KeyClass.tab, ["Tab"]],
  [KeyClass.backspace, ["Backspace"]],
  [KeyClass.delete, ["Delete"]],
  [KeyClass.navigation, ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"]],
  [
    KeyClass.modifier_alone,
    ["ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight", "MetaLeft", "MetaRight"],
  ],
];

const BY_CODE = new Map(GROUPS.flatMap(([keyClass, codes]) => codes.map((code) => [code, keyClass] as const)));

// Classed by the physical key, the same source as key_code.
export function keyClass(code: string): KeyClassCode {
  if (/^Key[A-Z]$/.test(code)) {
    return KeyClass.letter;
  }
  if (/^(Digit|Numpad)[0-9]$/.test(code)) {
    return KeyClass.digit;
  }
  if (/^F([1-9]|1[0-9]|2[0-4])$/.test(code)) {
    return KeyClass.function_key;
  }
  return BY_CODE.get(code) ?? KeyClass.other;
}
