## upload files
to upload files:
- **click on the file explorer.
select a file or shift+click multiple files**

the application can handle ```.pdf``` and ```.txt``` files.

## encoding text
this process reads text and translates it to numbers that the AI model can understand - words that are more similar are closer together numerically. \
("head" and "hat" are much more similar than "rhino" and "ocean")

- **enter the name of your encodings in the ```encodings name``` textbox.**
- **click the ```create document encodings``` button**

this will create a folder with a file ```embeddings.json``` inside. \
(inside this project's folder)

## loading encodings
if you've created encodings with this app before, you can skip straight to this step.
- **make sure the ```encodings name``` matches an encoding you've created**
- **click ```Load Document Encodings```**

## asking questions and generating answers
now that the encodings are loaded in,
- **type your question into the ```question``` box**
- **click ```generate response```**

you can do this as many times as you'd like - you do not need to reload the encodings each time.

to ask questions aobut new documents, redo ```upload files```, ```encoding text```, and ```loading encodings```,


# <span style="color:red">API key required to use Claude  </span>
in both the python file and jupyter notebook versions, you'll need an API key for claude. I am happy to share mine with you (just ask) or you can create your own at https://www.anthropic.com/api
