# Support material for the FAST SLR workflow

The workflow is based on the following stages:

- selection of bibliographic data set
- extraction of n-grams (automatic)
- classification of n-grams (manual)
- training of LDA using the keywords (automatic)
- validation of the topics obtained by the LDA (manual)
- clustering of papers
- derivation of statistics

# FA.WO.C. comments

## TODO and possible features

- Switch from CSV to TSV for internal operations (?).
- The input file should contain a column that, for each word, contains the reference to the papers containing the word.
  - Some words may refer to many many papers; probably it is better to have a different CSV containing the relationship word <-> paper.
- When a single word is classified, this should always open the list of n-grams containing the word; now this is supported for the words classified as "noise".
- NEW FEATURE: the program should be able to open a list of strings coming from the source abstracts that contain the word under exaxmination, to help understanding its relevance; this could be done only upon user request - if too computational heavy - or always in real-time - if not so heavy.
- The classification should save in the CSV file a meaningful word instead of a code (proposal: keyword, not-relevant, noise); empty fields is asssociated by default to unclassified/postponed words.
- Introduce the "relevant" class, which means that the word pertains to the domain evaluation, but it **may** not be used as keyword (e.g., it is too generic; example: "time" in real-time systems).
- Allow the user to configure the heys associated to the different actions/class; a simple JSON file may contain the association "key <-> action".
- Find an acronym for the program.
- Implement a graphical interface based on Tkinter.
- Split the logging between debug and useful info that can be used to profile the efficiency of the classification.
