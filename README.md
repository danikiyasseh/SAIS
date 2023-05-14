# SAIS

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


