export function choose(n, k) {
  if (k < 0 || k > n) return 0;
  const smallerSide = Math.min(k, n - k);
  let result = 1;
  for (let index = 1; index <= smallerSide; index += 1) {
    result = (result * (n - smallerSide + index)) / index;
  }
  return Math.round(result);
}
