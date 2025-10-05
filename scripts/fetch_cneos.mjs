// Fetch CNEOS data: Close Approaches (cad.api) and Sentry risk list (sentry.api)
// Outputs:
//   public/data/cneos/raw/close_approaches.json
//   public/data/cneos/raw/sentry_list.json
//   public/data/cneos/raw/sentry_removed.json
// Usage (PowerShell):
//   node scripts/fetch_cneos.mjs --date-min 2000-01-01 --date-max 2050-12-31 --limit 2000

import fs from 'node:fs'
import path from 'node:path'
import https from 'node:https'

const RAW_DIR = path.resolve('public', 'data', 'cneos', 'raw')
fs.mkdirSync(RAW_DIR, { recursive: true })

function get(arg, def) {
  const i = process.argv.indexOf(arg)
  if (i !== -1 && process.argv[i + 1]) return process.argv[i + 1]
  return def
}

const dateMin = get('--date-min', '2000-01-01')
const dateMax = get('--date-max', '2050-12-31')
const limit = Number(get('--limit', '1000'))

function fetchJSON(url) {
  return new Promise((resolve, reject) => {
    https
      .get(url, res => {
        let data = ''
        res.on('data', c => (data += c))
        res.on('end', () => {
          try {
            resolve(JSON.parse(data))
          } catch (e) {
            reject(new Error(`Failed JSON parse from ${url}: ${e}`))
          }
        })
      })
      .on('error', reject)
  })
}

async function fetchCloseApproaches() {
  // API docs: https://ssd-api.jpl.nasa.gov/doc/cad.html
  // Fields we care about: des, cd (date/time), dist (au), dist_min (au), v_rel (km/s), h (mag)
  const fields = ['des', 'cd', 'dist', 'dist_min', 'v_rel', 'h']
  let results = []
  let offset = 0
  while (true) {
    const url = `https://ssd-api.jpl.nasa.gov/cad.api?date-min=${dateMin}&date-max=${dateMax}&dist-max=0.5&sort=cd&limit=${limit}&offset=${offset}&fields=${fields.join(',')}`
    const json = await fetchJSON(url)
    const rows = json.data || []
    if (!rows.length) break
    results.push(...rows)
    offset += rows.length
    if (rows.length < limit) break
  }
  const out = {
    fields,
    count: results.length,
    data: results,
    meta: { dateMin, dateMax, fetchedAt: new Date().toISOString() },
  }
  fs.writeFileSync(path.join(RAW_DIR, 'close_approaches.json'), JSON.stringify(out, null, 2))
  console.log(`Saved ${results.length} close approaches -> public/data/cneos/raw/close_approaches.json`)
}

async function fetchSentryLists() {
  // API docs: https://ssd-api.jpl.nasa.gov/doc/sentry.html
  const current = await fetchJSON('https://ssd-api.jpl.nasa.gov/sentry.api?all=1')
  fs.writeFileSync(path.join(RAW_DIR, 'sentry_list.json'), JSON.stringify(current, null, 2))
  console.log(`Saved Sentry list (${current.count ?? current.data?.length ?? 0}) -> public/data/cneos/raw/sentry_list.json`)

  const removed = await fetchJSON('https://ssd-api.jpl.nasa.gov/sentry.api?removed=true')
  fs.writeFileSync(path.join(RAW_DIR, 'sentry_removed.json'), JSON.stringify(removed, null, 2))
  console.log(`Saved Sentry removed (${removed.count ?? removed.data?.length ?? 0}) -> public/data/cneos/raw/sentry_removed.json`)
}

async function main() {
  await fetchCloseApproaches()
  await fetchSentryLists()
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
