# Ideology and Power Identification in Parliamentary Debates 2024

This is a binary classification task, based on a speech of a speaker, to determine their belonging to coalition or opposition in the government.
[Link](https://touche.webis.de/clef24/touche24-web/ideology-and-power-identification-in-parliamentary-debates.html) to the task.
Used approaches:
* traditional ML;
* pretrained embeddings;
* instructed-tuned LLM.

## How to see our final report:
[Read](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/daschablume/power-identification/refs/heads/main/Power_classification_paper.pdf)

## How to run

First, you will need to install the required packages. Run this in the command line:

```bash
pip3 install -r requirements.txt
```

## Folder structure:
(To be updted)


## Country groups

- Balkan
  - Bosnia and Herzegovina (ba)
  - Croatia (hr)
  - Serbia (rs)

- Diff
  - Greece (gr)
  - Bulgaria (bg)

- Spanish
  - Spain (es)
  - Catalonia (es-ct)
  - Galicia (es-ga)
  - Basque Country (es-pv) [only power]

- Nordic
  - Denmark (dk) 
  - Finland (fi)
  - Iceland (is) [only political orientation] 
  - Norway (no) [only political orientation] 
  - Sweden (se) [only political orientation] 

- Slavic
  - Poland (pl)
  - Ukraine (ua)
  - Czechia (cz)
  - Serbia (rs)
  - Slovenia (si)

- West German
  - Austria (at)
  - Great Britain (gb)
  - The Netherlands (nl)
  - Norway (no) [only political orientation] 
  - Sweden (se) [only political orientation] 
  - Belgium (be)


- Romance
  - France (fr)
  - Portugal (pt)
  - Italy (it)

- Uralic
  - Estonia (ee) 
  - Hungary (hu)

- Baltic
  - Latvia (lv)
  - Lithuanian


- Turkic
  - Turkey (tr)
