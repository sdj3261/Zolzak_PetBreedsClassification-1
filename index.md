# **Graduation Project III**

## **🐶 Pet Breeds Classification 🐱**

<div style="text-align: right"> Gachon University, AI·Software department </div>
<div style="text-align: right"><strong> Author: Ahn Giju, Shin Dongjae </strong></div>
<div style="text-align: right"><strong> Advisor: Prof. Loh Woongkee </strong></div>

<br>

## Contents

- [Introduction](#introduction)
- [Development Environment](#development-environment)

## Introduction

저희 졸업 작품 프로젝트의 주제는 애완동물, 그 중에서도 '개와 고양이의 품종 분류하기' 입니다. 산책하다 마주친 멋진 강아지나, Youtube에서 봤던 귀여운 고양이의 품종이 궁금할 때가 있지요. 하지만 그 품종의 생김새를 묘사해서 검색한다는 것이 쉬운 일이 아닙니다. '얼굴과 발이 까만 고양이' ☜ 이런 식으로 검색해서 그 고양이의 품종을 찾아내기가 쉽지 않다는 것이죠. 그럴 때, 저희가 만든 이미지 분류 어플리케이션을 사용한다면 사진 한 장만 가지고도 손쉽고 재미있게 강아지 또는 고양이의 품종을 알아낼 수 있습니다.

저희는 강아지 또는 고양이 사진을 넣으면 품종이 무엇인지 알려주는 웹 베이스드 파이썬 프로그램을 만들었습니다. 나아가, 사용자가 자신의 얼굴 사진을 입력했을 때, 본인과 닮은 강아지 또는 고양이의 품종을 보여주는 기능(*닮은꼴 찾기)을 추가하여 앱을 사용하는 재미를 더했습니다.

The theme of our graduation project is classifying breeds of pets, especially dogs and cats. Sometimes I wonder what kind of cool dogs I encountered while taking a walk or what kind of cute cat I saw on Youtube. However, it is not easy to search by describing the appearance of the breed. "Cat with black face and feet" ☜ That's how it's hard to find the breed. In that case, if you use the image classification application we created, you can find out the breed of a dog or cat easily and interestingly with just one picture.

We created a web-based Python program that tells you what breed you have when you put pictures of dogs or cats. Furthermore, when a user enters a picture of his or her face, he or she adds a feature that shows the breed of a dog or cat that looks like him or her, adding to the fun of using the app.

아래 영상은 졸업작품II, 최종 프레젠테이션을 녹화한 것입니다. 프로젝트 진행 과정 및 결과를 전체적으로 설명하고 있으므로, 이 프로젝트에 관심이 있으시다면 시청을 추천합니다.

The video below is a recording of Graduation II, the final presentation. We're giving you a complete overview of the project progress and results, so if you're interested in this project, we recommend watching.

<p align="middle">
<iframe width="560" height="315" src="https://www.youtube.com/embed/nEeI4QNNJhY" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
</p>

## Development Environment

- Development Tool
  - Pycharm
  - Jupyter Notebook
  - Visual Studio Code

- Package manager
  - Anaconda 2019.03 version

- How to create a virtual environment and install packages for development
  1. Install [Anaconda](https://www.anaconda.com/).
  2. Download ['zolzak_conda_packages_export.txt'](https://github.com/GijuAhn/Zolzak_PetBreedsClassification/blob/gh-pages/zolzak_conda_packages_export.txt) here.
  3. Launch Anaconda prompt with administrator.
  4. You can run the commands below to create a virtual environment and install packages at once!

  ```cmd
  conda create -n(--name) VENV_NAME --file PATH\zolzak_conda_packages_export.txt
  ```

- Language
  - Python 3.7
  - Javascript
  - HTML/CSS/SCSS

- Mainly Used Libraries & Package
  - Numpy, Pandas, Scikit-learn, Matplotlib
  - Tensorflow 1.14.0, Tensorboard 1.14.0
  - Keras 2.1.5
  - Flask 1.1.2
  - OpenCV 4.3.0

- Dataset
  - [Stanford Dog dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
  - Google image crawling
  - Based on web crawling and intensive image preprocessing(automatically + manually), we have merged and replenished open source datasets to build custom datasets for our project. The dataset is sized as follows: 8351 images of 133 kinds of dogs, 4890 images of 30 kinds of cats.

## Idea Improvement

초기 목표는 하드디스크에 저장되지 않은 사진들이 정리되지 않은 채 마구잡이로 섞여 있는 사람들을 위한 object-detect 자동 사진 분류기, 앨범 정리 프로그램을 만들 계획이었습니다. 저희는 CIFAR-100 dataset (100개 클래스, 60000장 이미지) 으로 다양한 시도를 해 보았는데요, 여기서 superclass 의 개수가 많아질수록 웬만한 머신으로 커버할 수 없을 만큼 이미지 분류가 급격히 어려워진다는 사실을 파악했습니다. 여기서 superclass 의 개수를 줄이고, 대신 subclass를 세분화하여 접근하자는 피드백이 있었고, 이를 바탕으로 애완동물 품종 분류 애플리케이션이라는 아이디어로 발전시켰습니다.

The initial goal was to create an object-detect automatic photo classifier, album cleanup program for people whose photos were not stored on the hard disk and were mixed randomly. We've tried a variety of things with CIFAR-100 dataset (100 classes, 60000 images), and we've found that the more superclasses, the more difficult the image classification becomes, the more difficult it is to cover with most machines. There was a feedback here to reduce the number of superclasses, and instead to break down the subclasses, which led to the idea of pet classification applications.



2018년 농림축산검역본부의 통계[[기사]](http://www.animalrights.kr/news/articleView.html?idxno=823)에 따르면, 국내에서 반려동물을 기르는 가구 비율은 전체 가구의 23.7% 이고, 반려동물을 기르는 가정의 90%이상(개 75.9%, 고양이 14.4%)이 개 또는 고양이를 기르는 것으로 조사되었습니다. 따라서, 개와 고양이를 합리적인 수준으로 분류할 수 있다면, 전체 반려동물의 90% 이상을 커버할 수 있다고 판단했습니다.\
또한, 강아지는 많은 사람들이 키우는 만큼 그 종류와 수가 많지만, 고양이는 상대적으로 품종의 수도 적고, 키우는 사람들의 수도 적었습니다. 이 비율은 약 5:1 정도로 조사되었고, 이에 따라 개와 고양이 이미지 데이터셋을 각각 133종(8351이미지), 30종(4890이미지) 규모로 구축하였습니다.

According to statistics from the Ministry of Agriculture, Food and Rural Affairs in 2018, 23.7% of households raised pets in Korea, and more than 90% (75.9% of dogs, 14.4% of cats) raised dogs or cats. Therefore, if dogs and cats can be classified at a reasonable level, they can cover more than 90 percent of all pets.\
Also, dogs have many kinds and numbers as many people raise them, but cats have relatively few breeds and fewer people raise them. This ratio was found to be about 5:1, resulting in a dog and cat image dataset of 133 specifications to 8351 images and 30 specifications to 4890 images, respectively.

## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/GijuAhn/Zolzak_PetBreedsClassification/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/GijuAhn/Zolzak_PetBreedsClassification/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
