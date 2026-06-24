# IMSDb screenplay-to-scene parser

This command-line program downloads IMSDb screenplay HTML, extracts the screenplay `<pre>` block, sends the **complete screenplay** to the OpenAI Responses API, and writes one scene-character file per movie.

The LLM performs scene segmentation, character-presence extraction, and global alias normalization. Python validates the structured response and assigns integer IDs in first-appearance order.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_imsdb_parser.txt
```

Create a local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` and set your API key:

```dotenv
OPENAI_API_KEY=your-key
```

## Movie list

A plain-text list accepts any of these forms:

```text
Up|https://imsdb.com/scripts/Up.html
https://imsdb.com/scripts/American-Beauty.html
American Beauty
```

`Title|URL` is recommended because IMSDb filenames are not always predictable.

CSV is also supported:

```csv
title,url
Up,https://imsdb.com/scripts/Up.html
```

A CSV may use a `slug` column instead of `url`. A JSON array of strings or objects with `title`, `url`, or `slug` is also accepted.

## Run

```bash
python imsdb_llm_parser.py movies.example.txt \
  --output-dir movie_script_output
```

The default model is the reasoning model `gpt-5.5` with `medium` reasoning effort. Override the effort with:

```bash
python imsdb_llm_parser.py movies.example.txt \
  --output-dir movie_script_output \
  --reasoning-effort high
```

A lower-cost run can use:

```bash
python imsdb_llm_parser.py movies.example.txt \
  --output-dir movie_script_output \
  --model gpt-5.4-mini
```

Useful options:

```text
--reasoning-effort medium       none, low, medium, high, or xhigh
--overwrite-html                redownload cached screenplay HTML
--overwrite-json                rerun the LLM for existing output JSON
--keep-llm-output               retain the model's intermediate representation
--write-single-quote-json5      also write a single-quoted .json5 file
--api-attempts 3                structured-output retry count
--download-delay 1.0            delay between movies
--verbose                       detailed logging
```

## Output

```text
movie_script_output/
  html/          cached IMSDb pages
  json/          strict JSON files
  json5/         optional single-quoted JSON5 files
  llm_debug/     optional intermediate LLM output
  manifest.json  success/error summary
```

Strict JSON output uses two-space indentation and this schema:

```json
{
  "title": "Up",
  "character_ids": [
    {
      "name": "CARL",
      "id": 0
    }
  ],
  "scenes": [
    {
      "heading": "INT. MOVIE THEATRE - CONTINUOUS",
      "character_ids": [0]
    }
  ]
}
```

JSON requires double quotes. To also produce the requested two-space-indented, single-quoted representation, run:

```bash
python imsdb_llm_parser.py movies.example.txt \
  --output-dir movie_script_output \
  --write-single-quote-json5
```

That creates a `.json5` sidecar:

```javascript
{
  'title': 'Up',
  'character_ids': [
    {
      'name': 'CARL',
      'id': 0
    }
  ],
  'scenes': [
    {
      'heading': 'INT. MOVIE THEATRE - CONTINUOUS',
      'character_ids': [0]
    }
  ]
}
```

Characters include people who appear, act, speak, or are heard in the scene—not only dialogue speakers. Age variants such as `YOUNG CARL` and `CARL` are instructed to map to one canonical identity. Page numbers are not scenes; explicit screenplay sluglines are.

Use the downloader responsibly and comply with IMSDb's applicable terms and access policies.
