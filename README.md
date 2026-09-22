# faceid-bench

A measured attempt at a Face ID-like pipeline — detect, align, embed, match — built to be fast on
Apple hardware. Each model choice is backed by accuracy with confidence intervals and by latency
on the Apple Neural Engine (Core ML) and on a CUDA GPU.

RGB camera only: this project does not provide liveness or spoof resistance, which Face ID gets
from its infrared depth camera.

Work in progress.
