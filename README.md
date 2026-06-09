<div align="center">
  <img src="assets/logo.png" width="820" alt="Latent Spatial Memory logo" />

  <h1>Latent Spatial Memory for Video World Models</h1>

  <p align="center">
    <a href="https://arxiv.org/abs/2606.09828">
          <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&labelColor=111111" alt="Paper" />
    </a>
    <a href="https://aka.ms/latent-spatial-memory/">
          <img src="https://img.shields.io/badge/Project-Page-2ea44f?style=for-the-badge&labelColor=111111" alt="Project Page" />
    </a>
    <a href="https://github.com/microsoft/LatentSpatialMemory">
          <img src="https://img.shields.io/badge/Code-Coming%20Soon-555555?style=for-the-badge&logo=github&labelColor=111111" alt="Code" />
    </a>
  </p>

  <p>
    <a href="https://lhmd.top/">Weijie Wang</a><sup>1,*</sup> &nbsp;
    <a href="https://zhao-haoyu.github.io/">Haoyu Zhao</a><sup>1,*</sup> &nbsp;
    <a href="https://www.microsoft.com/en-us/research/people/yifanyang/">Yifan Yang</a><sup>2</sup> &nbsp;
    <a href="https://github.com/Chenfeng1271">Feng Chen</a><sup>3</sup> &nbsp;
    <a href="https://steve-zeyu-zhang.github.io/">Zeyu Zhang</a><sup>1</sup> &nbsp;
    <a href="https://hexy.tech/">Yefei He</a><sup>1</sup> &nbsp;
    <a href="https://zichengduan.github.io/">Zicheng Duan</a><sup>3</sup><br/>
    <a href="https://donydchen.github.io/">Donny Y. Chen</a><sup>4</sup> &nbsp;
    <a href="https://www.microsoft.com/en-us/research/people/yuqyang/">Yuqing Yang</a><sup>2</sup> &nbsp;
    <a href="https://bohanzhuang.github.io/">Bohan Zhuang</a><sup>1</sup>
  </p>

  <p>
    <sup>1</sup>Zhejiang University &nbsp;&nbsp;
    <sup>2</sup>Microsoft Research &nbsp;&nbsp;
    <sup>3</sup>Adelaide University &nbsp;&nbsp;
    <sup>4</sup>Monash University
  </p>

  <p><sup>*</sup>Equal contribution</p>
</div>

<p align="center">
  <img src="assets/concept.png" width="900" alt="Latent Spatial Memory concept" />
</p>

<p align="center">
  <b>Latent Spatial Memory stores persistent 3D scene content directly as latent tokens.</b><br />
  It avoids repeated RGB rendering and re-encoding from explicit 3D caches, enabling efficient spatial consistency for video world models.
</p>

## Highlights

<table>
  <tr>
    <td align="center"><b>Latent Memory</b><br />Persistent 3D scene context lives directly in latent space.</td>
    <td align="center"><b>No RGB Detour</b><br />Mirage avoids repeated render-and-reencode cache updates.</td>
  </tr>
  <tr>
    <td align="center"><b>Memory Lifecycle</b><br />Initialize, read, denoise, and update across generated chunks.</td>
    <td align="center"><b>Efficient Worlds</b><br />Higher generation efficiency with lower 3D cache memory.</td>
  </tr>
</table>

## Method

<p align="center">
  <img src="assets/architecture.png" width="900" alt="Mirage architecture" />
</p>

<p align="center">
  Mirage builds a persistent latent cache from the initial observation. For each generated chunk, it reads target-view memory, uses it during denoising, and writes updated static scene content back to the cache.
</p>

## Results

<p align="center">
  <img src="assets/efficiency.png" width="900" alt="Mirage efficiency results" />
</p>

<table align="center">
  <tr>
    <td align="center"><h3>10.57x</h3>faster generation</td>
    <td align="center"><h3>55x</h3>lower 3D cache memory</td>
    <td align="center"><h3>70.36</h3>WorldScore average</td>
  </tr>
</table>

## Citation

If you find this project useful, please cite:

```bibtex
@article{wang2026mirage,
  title   = {Latent Spatial Memory for Video World Models},
  author  = {Wang, Weijie and Zhao, Haoyu and Yang, Yifan and Chen, Feng and Zhang, Zeyu and He, Yefei and Duan, Zicheng and Chen, Donny Y. and Yang, Yuqing and Zhuang, Bohan},
  journal = {arXiv preprint arXiv:2604.24764},
  year    = {2026}
}
```

## Support

See [SUPPORT.md](SUPPORT.md) for usage support and [SECURITY.md](SECURITY.md) for vulnerability reporting.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution.

For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com). When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately. Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
