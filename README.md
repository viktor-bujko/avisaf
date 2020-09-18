## Avisaf
Avisaf is a tool used for **creating**, **training** and **testing** spaCy-based
Named Entity Recognition models in Aviation Safety Reports.

### Installation 

Avisaf is now available primarily for Unix-like OS:

Please run `$ ./install.sh` script which installs the application and also 
downloads spaCy default English language model.
     
### How to launch the application

* Application can be started by typing `avisaf` + one of the following 
commands: `test`, `train`, `autobuild`, `build`.  
Example: `avisaf test [additional arguments]`
    
* Feel free to add `-h` parameter anytime if you're in doubt about how
the program should be used.

### Documentation

Read more details about avisaf in the documentation available [here].

[here]:http://www.ms.mff.cuni.cz/~bujkov/avisaf/index.html
     
### Program description

The program consists of 3 different NER model creation steps which 
will be described below:

* avisaf \[auto]build
* avisaf train
* avisaf test
    
#### \[auto]build
Both **build** and **autobuild** are subcommands responsible for **creation** of 
annotated training data. The goal is to create a list of (start, end, label) tuples,
where each tuple describes starting and ending indexes of an entity as well as the 
label attributed to the entity. Only then may such annotated data be used for new 
model training.   
  
Automatic annotation (autobuild) makes use of spaCy [Matcher] and [PhraseMatcher] 
objects. Firstly, texts dedicated for annotation have to be loaded either as raw 
strings without any annotation from a CSV file or as tuples (from JSON file) which 
contain a text string and a dictionary holding the set of existing entities under 
"entities" key. Entities recognition in autobuild is based on rules matching, 
which means that the rules (referred to as patterns) have to be loaded from a user 
given pattern-file as well. This [link] provides more details and information 
about rule-based matching and pattern formats. After that, the patterns are looked 
for in each text, and for every match, starting index, ending index and user 
provided label are used for tuple creation. However, one additional step needs 
to be done before saving the final result. Since each token may be part of only 
one entity and the technique described above may create entity overlaps, such 
conflicts must be resolved. Overlap resolution is done by sorting the entities 
by their starting index, and then deleting shorter entity. Even though such 
solution is not perfect, it provides quite satisfactory results most of the time.
Finally, the new data can be saved or printed based on user preference and be 
used for training purposes.      
  
[Matcher]: https://spacy.io/api/matcher
[PhraseMatcher]: https://spacy.io/api/phrasematcher
[link]: https://spacy.io/usage/rule-based-matching#matcher

Manual annotation (build) leads to the same result as above despite several 
differences. Firstly, manual annotation can only be performed on the texts from 
a CSV file (path given as parameter) or on the texts written by user to the 
standard input. When extracting the texts from CSV, the user is able to limit the 
number of extracted texts by providing the start index and the number of lines 
to be read. Then, each of these texts is printed to the standard output and the 
user is prompted to write all the words he/she wants to annotate. He/she is given
the list of all available entity labels or 'NONE' if a word was typed in 
accidentally. After all the texts were annotated, the entity list is cleaned 
from overlaps and saved or printed as above.
     
#### train
TBD

#### test
TBD

### Obtained results

TBD

The results can be checked either by checking the recognition accuracy `stats checker`
or by `avisaf test` and the result visualization.

### Examples

![NER example 1](results/example1.png)

![NER example 2](results/example2.png)

&copy; Viktor Bujko 2020
