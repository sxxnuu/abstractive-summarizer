from transformers import pipeline

# 영어 요약 모델 
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
A volcano in Ethiopia has erupted for the first known time in 10,000 years, spewing plumes of thick smoke and ash high into the sky and impacting air travel thousands of miles away in India.
The long-dormant Hayli Gubbi volcano in the Afar region in Ethiopia’s northeast roared to life Sunday, covering the neighboring villages in dust and creating challenges for farmers.
While no casualties were reported, the eruption poses a threat to the local community of livestock herders by smothering vital grazing lands, local administrator Mohammed Seid told The Associated Press
Residents described hearing a terrifying blast at the moment of the eruption.
“It felt like a sudden bomb had been thrown with smoke and ash,” local resident Ahmed Abdela told the news agency.
The eruption was visible from satellites, with NASA images showing thick plumes of dust rising into the sky and billowing across the Red Sea.
Volcanic clouds from the eruption drifted over Yemen, Oman, and into Pakistan and India, according to the Toulouse Volcanic Ash Advisory Center.
Pakistan’s Meteorological Department issued a warning after ash entered its airspace late on Monday.
In India, flag carrier Air India cancelled several domestic and international flights to carry out “precautionary checks on those aircraft which had flown over certain geographical locations after the Hayli Gubbi volcanic eruption,” it said on X.
Delhi, which is experiencing a wave of severe air pollution, is not expected to be significantly affected because the ash is drifting at a high altitude, India’s Meteorological Department (IMD) said.
The plumes are expected to rapidly move eastwards, the IMD added.
Located about 800 kilometers (500 miles) northeast of capital Addi Ababa, Hayli Gubbi is the southernmost volcano of the Erta Ale Range, a volcanic chain in Ethiopia’s Afar region.
It rises about 500 meters in altitude and sits within a zone of intense geological activity where two tectonic plates meet.
"""

summary = summarizer(
    article,
    max_length=150,   # 요약 최대 길이
    min_length=40,    # 요약 최소 길이
    do_sample=False   # 랜덤성 제거 → 안정적인 요약
)[0]['summary_text']

print("\n📌 요약 결과:\n")
print(summary)
