# Language Recognizer
Given a text, examine the script and word formations to determine the language. 
The aim is to correctly distinguish those languages that follow same script.
Meant to be used as the fist step for text processing pipelines that use language dependant next steps like spell checks or rendering.

Technology: Python

Methodology: 
 1. Determine the script used.
 2. For those scripts, that are followed by multiple languages;
    have a simple ML model that does text classification
    based on simple features like presence of common words or frequent n-grams.

Usage:

> Coming soon!