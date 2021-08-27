# StyleGAN, InterfaceGAN, IDInvert and FaceAttributedFAN

This project was done by (all contributed the same): Brayam Castillo, Filipe Lauar, Gabriel Baker and Vinícius Imaizumi

More details in the file GAN_report.pdf. Implementation in GAN_image_manipulation.ipynb. You should run the [code on google colab](https://colab.research.google.com/drive/1qVoqCA2i62PdcO2Pdfiv944mxHVN8zxt)

Project to use [StyleGAN](https://github.com/NVlabs/stylegan), [InterfaceGAN](https://github.com/genforce/interfacegan), [IDInvert](https://github.com/genforce/idinvert_pytorch) and [FaceAttributeFAN](https://github.com/TencentYoutuResearch/FaceAttribute-FAN) to manipulate images in StyleGAN's latent space.

What was done:

- Combined the style of two real persons with StyleGAN Changed different facial attributes using InterfaceGAN.

- Used IDInvert from Genforce (https://github.com/genforce/idinvert_pytorch) to map real images into the StyleGAN latent space W. 

- Interpolated the code from real people in this latent space W, achieving a person with intermediate face characteristics between the two images used.

- Used InterfaceGAN increase or decrease a specific but abstract attribute, such as smile, age and gender.

- Used FaceAttributeFAN to see other features that changed together (ie: entangled features)


The explanation of the network achitecture and more are in my podcast:

- [What's behing instagram filters? Understanding the styleGAN.](https://open.spotify.com/episode/0JvrnMNBOYjqDUYzDXqm3F?si=25f792058ed548e8) (In english)
- [O que está por tras dos filtros do instagram? Explicando a styleGAN.](https://open.spotify.com/episode/5u5wnPx2Pb9ZsdGsyWaCHi?si=ee6ef00550bc409c) (Em portugues)
