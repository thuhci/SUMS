# SUMS[UIC'24]
Summit Vitals: Multi-Camera and Multi-Signal Biosensing at High Altitudes

## 📖 Abstract
Here is SUMS dataset collected by Qinghai University. It is a multi-Camera and multi-Signal Biosensing dataset collected at high altitudes, which includes 80 synchronized non-contact facial and contact finger videos from 10 subjects during exercise and oxygen recovery scenarios. This dataset captures PPG, respiration rate (RR), and SpO2, and is designed to validate video vitals estimation algorithms and compare facial rPPG with finger cPPG. Our results demonstrate that fusing videos from different positions (face and finger) reduces the mean absolute error (MAE) of SpO2 predictions by 7.6% and 10.6% compared to using only face or only finger data. Additionally, training on multiple indicators such as PPG and blood oxygen simultaneously reduces SpO2 estimation MAE by 17.8%.

## 🔍 Experiment Setup
We recruited ten participants living on the Qinghai Plateau to collect hypoxia data in a real high-altitude environment. Data collection utilized two Logitech C922 cameras to capture videos of participants’ faces and fingers, along with a CMS50E+ oximeter for measuring PPG signals and blood oxygen levels. Respiratory patterns were monitored using an HKH-11C respiratory sensor. Video recordings were captured at a resolution of 1280x720 pixels, running at 60 frames per second. The PPG signals were recorded at a frequency of 20 Hz, while respiratory waves were captured at 50 Hz. The experiment setup is shown as follows.

![device](https://github.com/user-attachments/assets/061bf60a-a9a3-498e-9f26-cf21e460b5b1)

(Schematic illustration of the experimental setup used for data collection while participants are having oxygen inhalation.)

## ⚙️ Experiment Procedure
The experimental process involved two sets of experiments, each with a rest state and a recovery state. In the first set, conducted in the morning, subjects wore a breathing sensor (HKH-11C) on their abdomen, an oximeter (CMS50E+) on their left index finger, and positioned their right index finger on a camera (Logitech C922). Data was recorded for three minutes during the rest state (state 1). Afterward, they performed a stair-climbing exercise, followed by another recording session (state 2) for two minutes.

The second set of experiments took place in the afternoon after subjects had fully recovered. The setup was similar, with a three-minute rest period (state 3), followed by another stair-climbing exercise. This time, during the post-exercise recording (state 4), which lasted two minutes, subjects wore portable oxygen devices.

![Experiment process](https://github.com/user-attachments/assets/bc32a475-d392-4734-8784-f101b0536976)

(A visual illustration of our data collection protocol)

## 🗝️ Access and Usage
**This dataset is built for academic use. Any commercial usage is banned.**   
To access the dataset, you are supposed to download this [data release agreement](https://github.com/thuhci/SUMS/blob/main/SUMS_Release_Agreement.pdf).  
Please scan and dispatch the completed agreement via your institutional email to <tjk24@mails.tsinghua.edu.cn> and cc <yuntaowang@tsinghua.edu.cn>. The email should have the subject line 'SUMS Access Request -  your institution.' In the email,  outline your institution's website and publications for seeking access to the SMUS, including its intended application in your specific research project. The email should be sent by a faculty rather than a student.   

## Citation  
Title: [Summit Vitals: Multi-Camera and Multi-Signal Biosensing at High Altitudes](https://arxiv.org/abs/2409.19223)  
Ke Liu*, Jiankai Tang*(*Co-first Author), Zhang Jiang, Yuntao Wang, Xiaojing Liu, Dong Li, Yuanchun Shi, "Summit Vitals: Multi-Camera and Multi-Signal Biosensing at High Altitudes", IEEE UIC, 2024  
```
@misc{liu2024summitvitalsmulticameramultisignal,
      title={Summit Vitals: Multi-Camera and Multi-Signal Biosensing at High Altitudes}, 
      author={Ke Liu and Jiankai Tang and Zhang Jiang and Yuntao Wang and Xiaojing Liu and Dong Li and Yuanchun Shi},
      year={2024},
      eprint={2409.19223},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.19223}, 
}
```
