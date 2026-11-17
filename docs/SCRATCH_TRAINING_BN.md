# AIBondhu — শূন্য থেকে CPU প্রশিক্ষণ

এই পথটি কোনো কোম্পানির pretrained weights বা pretrained tokenizer ব্যবহার করে না। Transformer-এর সব parameter random initialization থেকে শুরু হয়; tokenizer-ও নির্বাচিত corpus থেকে নতুন করে train হয়।

## GitHub Actions
Actions → **AIBondhu Scratch CPU 5h** → Run workflow। Workflow CPU-only এবং সর্বোচ্চ 300 মিনিটের training window চায়; runner setup/download/evaluation/artifact-এর জন্য অতিরিক্ত সময় রাখা হয়েছে। GitHub-hosted runner-এর সময়সীমা, quota, availability এবং repository billing policy প্রযোজ্য। Job থেমে গেলে সম্পূর্ণ ৫ ঘণ্টা training হয়েছে বলা যাবে না।

## Corpus
Workflow PMC Open Access থেকে CC0, CC BY, CC BY-SA লাইসেন্সের নিবন্ধ সংগ্রহ করে। এই corpus প্রধানত biomedical text; তাই মডেলটি সাধারণ জ্ঞানের সর্ববিষয়ক AI নয়। Attribution manifest artifact-এ থাকে। কোনো patient record বা ব্যক্তিগত স্বাস্থ্যতথ্য যোগ করা যাবে না।

## সীমাবদ্ধতা
শূন্য থেকে ৫ ঘণ্টা CPU training করে অত্যন্ত বুদ্ধিমান বা চিকিৎসাগতভাবে নির্ভুল LLM তৈরি করা বাস্তবসম্মত নয়। এই প্রকল্পটি একটি ছোট গবেষণা-প্রোটোটাইপ। Loss/perplexity কেবল পরবর্তী-token prediction মাপে, চিকিৎসা-নিরাপত্তা নয়। Diagnosis, triage, prescription, dosage বা রোগীর চিকিৎসায় ব্যবহার নিষিদ্ধ; clinician-reviewed evidence retrieval ও স্বাধীন clinical validation ছাড়া কোনো deployment নয়।
