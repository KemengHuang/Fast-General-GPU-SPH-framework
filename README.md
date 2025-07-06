# Fast-SPH-framework
This framework represents a fast general implementation of GPU SPH method utilizing the uniform grid approach.

DESCRIPTION
===========
This project is the source code of ["Novel Hierarchical Strategies for SPH-centric Algorithms on GPGPU"](https://doi.org/10.1016/j.gmod.2020.101088)
and ["A General Novel Parallel Framework for SPH-centric Algorithms"](https://dl.acm.org/doi/10.1145/3321360). 

This project offers fast optimization strategies based on uniform grid. When compared to a well-optimized GPU SPH method based on the uniform grid, the method proposed in the papers demonstrates a significant speed improvement of up to 3.5 times. As a result, it serves as an excellent benchmark for conducting further research on GPU SPH and facilitates meaningful comparisons.


Source code contributor: [Kemeng Huang](https://kemenghuang.github.io), Jiming Ruan

**Note: this software is released under the MPLv2.0 license. For commercial use, please email authors for negotiation.**

## BibTex 

Please cite the following papers if it helps. 


```
@article{HUANG2020101088,
  title = {Novel hierarchical strategies for SPH-centric algorithms on GPGPU},
  journal = {Graphical Models},
  volume = {111},
  pages = {101088},
  year = {2020},
  issn = {1524-0703},
  doi = {https://doi.org/10.1016/j.gmod.2020.101088},
  url = {https://www.sciencedirect.com/science/article/pii/S152407032030028X},
  author = {Kemeng Huang and Zipeng Zhao and Chen Li and Changbo Wang and Hong Qin}
}
```


```
@article{10.1145/3321360,
  author = {Huang, Kemeng and Ruan, Jiming and Zhao, Zipeng and Li, Chen and Wang, Changbo and Qin, Hong},
  title = {A General Novel Parallel Framework for SPH-Centric Algorithms},
  year = {2019},
  issue_date = {May 2019},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {2},
  number = {1},
  url = {https://doi.org/10.1145/3321360},
  doi = {10.1145/3321360},
  journal = {Proc. ACM Comput. Graph. Interact. Tech.},
  month = {jun},
  articleno = {7},
  numpages = {16}
}
```
