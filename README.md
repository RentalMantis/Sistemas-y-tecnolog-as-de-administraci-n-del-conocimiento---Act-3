# Sistemas-y-tecnolog-as-de-administraci-n-del-conocimiento---Act-3
Implementation of a Python script to generate top-K document recommendations based on cosine similarity, while validating its results and business utility.
# Document Recommender (TF-IDF + Cosine Similarity)

## Objective

This project is a Python program that finds the 5 documents most similar to a text. It works with a set of 30 made-up company documents: IT tickets, HR policies, meeting notes, FAQs, security policies, and customer support cases.

You paste a short text, and the program shows the 5 best matches with a score for each one. You can also use a document ID instead of a text. In that case, the document itself is not included in its own results.

**How it can be used:** to suggest the right policy or FAQ to an employee with a question, or to show related tickets to a support agent.

**How it works**

1. Clean the text: make it lowercase and remove symbols and common words (like "the" and "and").
2. Turn each document into numbers with TF-IDF, a score that shows which words matter most in each document.
3. Compare the numbers with cosine similarity, a score from 0 to 1. A higher score means the texts are more alike.
4. Show the 5 best matches.

## Dependencies

- Python 3.9 or higher
- pandas
- scikit-learn (it installs NumPy automatically)

## Project files

| File | Description |
|`Evidences`| Screenshots of the working program |
| `recommender.py` | The program |
| `corpus.csv` | The 30 documents, with the columns `id`, `title`, `text`, and `category` |
| `README.md` | This file |

## Execution steps

1. Save `recommender.py` and `corpus.csv` in the same folder.
2. Open a terminal in that folder.
3. Install the dependencies.
4. Run the program:

   ```bash
   python recommender.py


5. Paste a short text, press Enter, and then press Enter again on the empty line.
6. Read the 5 results. Paste another text, or type `q` to quit.

## Usage example

Evidences/recommender test 2.png

## Notes

- The program finds matches by shared words, not by meaning. For example, "work from home" may not match a document that says "remote access".
- A document with a score of 0 is not shown, so you may sometimes see fewer than 5 results.