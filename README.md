# PDF Question Answering Tool

A question answering tool that reads PDF files and answers natural language questions using only the text inside them. Built for **Assignment 2: NLP** by **Tarek Alam Bhuiyan (ID: 1000129291)**.

The tool was tested on two PDFs about Bangladesh: an **Easy** PDF with clean text, and a **Hard** PDF where long blocks of junk symbols (`$ % & ( ) _`) are inserted inside sentences. The questions are read automatically from a third PDF.

## How It Works

```
PDFs -> text extraction -> noise cleaning -> passage splitting -> TF-IDF retrieval -> extractive QA model -> answer
```

1. **Extraction:** text is pulled from every page with `pypdf`.
2. **Cleaning:** a regex removes runs of noise symbols, then line breaks and extra spaces are normalised so broken sentences are rejoined.
3. **Passages:** text is split into sentences and grouped into overlapping 3-sentence windows, so answers that cross a sentence boundary are still found. Duplicate passages across PDFs are dropped.
4. **Retrieval:** TF-IDF (unigrams and bigrams) with cosine similarity selects the top 3 passages for each question.
5. **Answer extraction:** the pretrained model [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2) predicts the start and end tokens of the answer in each passage. The span with the highest confidence is returned.
6. **Question loading:** `Questions.pdf` is parsed by splitting on question marks and removing numbering. Line 6 holds two questions, so 7 questions are answered in total.

The notebook also includes an interactive widget where you can type any question and see the answer, a confidence badge and the supporting passage.

## Results

All 7 questions were answered correctly on the Easy PDF, the Hard PDF and both PDFs combined.

| No | Question | Answer |
|----|----------|--------|
| 1 | What is the capital city of Bangladesh? | Dhaka |
| 2 | Which river is considered the lifeline of Bangladesh? | Padma River |
| 3 | In which year did Bangladesh gain independence? | 1971 |
| 4 | What is the official language of Bangladesh? | Bengali |
| 5 | What is the name of the world's largest mangrove forest located in Bangladesh? | Sundarbans |
| 6 | Which currency is used in Bangladesh? | Bangladeshi Taka |
| 7 | What is the national animal of Bangladesh? | Royal Bengal Tiger |

The Hard PDF gave the same answers as the Easy PDF, which shows the noise cleaning works. Confidence scores ranged from 0.795 to 0.993.

## Getting Started

### Requirements

- Python 3.9+
- `pypdf`, `transformers`, `torch`, `scikit-learn`, `pandas`, `ipywidgets`

```bash
pip install pypdf transformers torch scikit-learn pandas ipywidgets
```

The first run downloads the model (about 500 MB), so an internet connection is needed.

### Run in Google Colab

1. Open `Assignment_2_PDF_QA_Tool_Tarek_Alam_Bhuiyan_1000129291.ipynb` in Colab.
2. Upload the three PDFs to `/content/`.
3. Check the paths in the **Config** cell:

   ```python
   EASY_PDF = "/content/Test_PDF_Easy.pdf"
   HARD_PDF = "/content/Test_PDF_Hard.pdf"
   QUESTIONS_PDF = "/content/Questions.pdf"
   ```
4. Run all cells.

### Run locally

Install the requirements, start Jupyter, open the notebook, change the three paths in the Config cell to your local files, and run all cells.

### Use with your own PDFs

```python
tool = PDFQuestionAnswerer(["your_file.pdf"])
result = tool.answer("Your question here?")
print(result["answer"], result["score"])
```

## Configuration

| Setting | Default | Meaning |
|---------|---------|---------|
| `MODEL_NAME` | `deepset/roberta-base-squad2` | Extractive QA model |
| `TOP_K` | 3 | Passages passed to the QA model |
| `WINDOW` | 3 | Sentences per passage |
| `STRIDE` | 1 | Sentence step between passages |

## Limitations and Future Work

- The tool is **extractive**, so answers are short spans copied from the PDFs.
- Scanned PDFs are not supported because there is no OCR step.
- Possible improvements: dense retrieval with `sentence-transformers`, OCR with `pytesseract`, and a generative reader for longer answers.

## Tech Stack

Python, pypdf, scikit-learn, Hugging Face Transformers, PyTorch, pandas, ipywidgets.

## Author

**Tarek Alam Bhuiyan**
GitHub: [tarek-codes](https://github.com/tarek-codes)
