export function isChinaNetwork(): boolean {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const chinaZones = [
      'Asia/Shanghai',
      'Asia/Chongqing',
      'Asia/Urumqi',
      'Asia/Harbin',
      'Asia/Kashgar',
    ];
    return chinaZones.includes(tz);
  } catch {
    return false;
  }
}