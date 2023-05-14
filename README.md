# SAIS

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

SAIS is a surgical AI system that decodes surgical activity based exclusively on surgical videos. This system was first introduced in a 2023 [Nature Biomedical Engineering](https://www.nature.com/articles/s41551-023-01010-8) paper. 

## Features

SAIS has the ability to decode the following elements of surgery:
- [x] surgical steps (e.g., dissection vs. suturing)
- [x] suturing sub-phases (e.g., needle handling vs. needle driving vs. needle withdrawal)
- [x] four distinct suturing gestures   
- [x] six distinct tissue dissection gestures
- [x] needle handling binary skill level  
- [x] needle driving binary skill level 

## Inference

To demonstrate the utility of SAIS, we have developed an [interface](https://huggingface.co/spaces/danikiyasseh/SAIS/tree/main) that allows users to upload a brief surgical video (on the order of 10-20 seconds) and identify:
- six distinct tissue dissection gestures
- binary skill level of suturing activity


