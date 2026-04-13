## Babel Elephant

This is a [submission](https://devpost.com/software/hacksmu-vii-elephant-vocalization-project) to the HackSMU VII hackathon for the [iMasons-Elephant Voices challenge](https://web.archive.org/web/20260413120429/https://docs.google.com/document/d/13aKidfok8dGQUB0N5Y_kNwhLNiGvsvarodmbqxSKA8E/edit?tab=t.0#heading=h.u1wl1z5r6qo3) which ran from April 11-12, 2026. The challenge statement was:

> Develop a method to remove overlapping noise from elephant recordings (generators, cars, planes) without distorting the elephant call.
> * Use spectrograms to visualize sound
> * 44 audio recordings with 212 elephant calls embedded in mechanical noise
> * Some of the calls overlap with each other too
> * Spreadsheet has sound file name, start time, and end time for each call

We implemented a convolutional neural network to denoise elephant rumble audio files, based on [existing academic research](https://www.mdpi.com/2571-5577/7/6/117) on the subject.

### AI Disclosure

Considering the short 24-hour time constraint of HackSMU VII, we used off-the-shelf LLMs such as Claude Code and Perplexity to assist with some of the implementation and debugging.
