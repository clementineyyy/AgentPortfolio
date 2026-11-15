import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import test from "node:test"

const pageSource = await readFile(new URL("./page.tsx", import.meta.url), "utf8")

test("portfolio benchmark is initialized once and cannot reset accepted data on cash changes", () => {
  assert.match(pageSource, /useState<PortfolioState>\(\(\) => createBenchmarkData\(1_000_000\)\)/)
  assert.doesNotMatch(pageSource, /useEffect\s*\([\s\S]*?setCurrentState\s*\(/)
})
