const CONFIG = {
      "appTitle": "COVID LAT Annotator",
      "adminEmails": ["agglutinationtest@gmail.com"],
      "sheetsConfig": {
            "API_KEY": "AIzaSyDY63oaWE0SEcgPL2Hf1hSz8wWLwlYqd6Q",
            "CLIENT_ID": "509563213429-46knn18d8v6krn02vcn2merag004212d.apps.googleusercontent.com",
            "SCOPE": "https://www.googleapis.com/auth/spreadsheets"
      },
      "users": {
            "default": {
                  "SPREADSHEET_ID": "1UBH-A4dQcyGGN1kf3BgPVZDThV7US15MJ3f2udJf5ek",
                  "SPREADSHEET_RANGE": "COVID_Annotations",
                  "URL_SUBSTRINGS_TO_ANNOTATE": ['20210727_15_02'],
                  "FRAME_TYPE_TO_ANNOTATE": 'mod_5'
            }
      },
      "annotationRules": {
            "0 - No evidence of agglutination": [
                  "Nothing, or only cloudiness"
            ],
            "1 - Any evidence of agglutination": [
                  "Small specs",
                  "Cloudiness where the background breaks up into different regions (but there must be specs)"
            ],
            "2 - Evidence of agglutination": [
                  "Getting darker and more pronounced",
                  "Still have blobs the are scattered",
            ],
            "3 - Definite agglutination": [
                  "Textbook definition: a few large agglutinates",
                  "Scattered blobs look like they are starting to connect",
                  "A few large dark blobs. The more the spread, the more likely it is a 2",
                  "Dark blobs, but cloudiness in the background"
            ],
            "4 - Strong agglutination": [
                  "Strong dark blob",
                  "No cloudiness in the background",
                  "Use cloudiness in the background to distinguish between a 3 & 4"
            ]
      },
      "imgHeight": 500,
      "imgWidth": 500,
      "referenceList": {
            "0":[
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_0_5_11.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_2_4_8.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_1_5_8.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_2_4_9.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_0_5_10.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_0_5_12.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_2_4_10.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_2_4_11.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_0_5_8.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_0_5_9.png",
            ],
            "1":[
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_11_58_B7_35.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_E6_50.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_C2_50.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_B2_65.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_E1_110.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_E2_30.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_D9_115.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_E2_100.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_C6_90.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_B2_75.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_C8_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_E7_105.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_D1_30.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_D2_45.png"
            ],
            "2":[
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_E3_120.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_C1_110.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_11_58_B10_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_11_58_C4_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_G5_120.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_17_37_D9_50.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_17_37_G2_59.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210604_18_08_B4_55.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210604_18_08_B5_25.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_11_58_B8_40.png"
            ],
            "3":[
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_C8_110.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_11_58_D5_55.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_D4_65.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_C8_90.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_E5_110.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_11_58_C12_60.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_G7_85.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_F7_75.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_11_58_D6_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210624_13_46_D8_85.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_11_58_B6_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_17_37_G9_50.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_17_37_G8_55.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_17_37_E6_59.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210603_11_58_D12_60.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210604_18_08_D9_60.png"
            ],
            "4":[
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210604_13_12_C4_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210604_13_12_C4_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_G9_50.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_C11_30.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_D11_55.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_D9_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_B10_55.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_B8_50.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_G8_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_F10_35.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_F10_55.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_D10_35.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_C11_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_E11_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_G10_45.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_B8_55.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/20210622_17_33_E10_45.png"
            ],
            "NA":[
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_2_7_19.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_2_7_26.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_1_7_19.png",
               "https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/2021-04-30_rabbit_0_7_19.png"
            ]
      }
}
export default CONFIG