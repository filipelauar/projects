More details in the file GAN_report.pdf. Implementation in GAN_image_manipulation.ipynb.

Project to use [StyleGAN](https://github.com/NVlabs/stylegan), [InterfaceGAN](https://github.com/genforce/interfacegan), [IDInvert](https://github.com/genforce/idinvert_pytorch) and [FaceAttributeFAN](https://github.com/TencentYoutuResearch/FaceAttribute-FAN) to manipulate images in StyleGAN's latent space.

What was done:

- Combined the style of two real persons with StyleGAN Changed different facial attributes using InterfaceGAN.

- Used IDInvert from Genforce (https://github.com/genforce/idinvert_pytorch) to map real images into the StyleGAN latent space W. 

- Interpolated the code from real people in this latent space W, achieving a person with intermediate face characteristics between the two images used.

- Used InterfaceGAN increase or decrease a specific but abstract attribute, such as smile, age and gender.

- Used FaceAttributeFAN to see other features that changed together (ie: entangled features)