---
license: other
license_name: mixed-cc
license_link: https://creativecommons.org/licenses/
task_categories:
- text-to-image
language:
- en
tags:
- architecture
- islamic-architecture
- mamluk
- cairo
- lora
- flux
pretty_name: ArchForge - Mamluk / Islamic Cairo Architecture
size_categories:
- n<1K
---

# ArchForge — Mamluk / Islamic Cairo Architecture Dataset

A small, curated, licence-documented image dataset of **Mamluk and Islamic Egyptian
architecture**, built for LoRA style adaptation on FLUX.1-dev. 39 images at
512x512px, 33 train / 6 validation.

Built as part of [ArchForge](https://github.com/marwantosolve/ArchForge) — a time-boxed proof of concept,
not a production dataset. The evaluation, the LoRA adapter and the comparison grid
are linked from there.

## Why this dataset exists

The target is a *style*, not a subject: carved limestone facades, pointed arches,
muqarnas vaulting, mashrabiya screens, ablaq striped stonework, domes and minarets
of historic Cairo. The curation work was therefore mostly **rejection** — Wikimedia
Commons categories mix architecture with book scans, museum objects, architectural
plans, paintings, lithographs, and buildings from other countries entirely.

## Provenance and licensing

Every image comes from **Wikimedia Commons** and retains its original licence.
No image was scraped from a source without a checkable licence, and no licence was
inferred. Attribution for each file is recorded in `metadata.csv` and listed below.

| Licence | Images |
|---|---|
| CC BY-SA 4.0 | 14 |
| CC BY-SA 3.0 | 14 |
| CC BY 3.0 | 4 |
| CC BY-SA 2.0 | 3 |
| CC BY 2.0 | 2 |
| Public domain | 1 |
| No restrictions | 1 |

**Reuse obligation:** most files are CC BY or CC BY-SA, which require attribution
and (for SA) share-alike. If you reuse this dataset, carry the per-image attribution
in the table below. Public domain and "no restrictions" files carry no such condition.

## Curation method

1. **Collection** (`scripts/collect_dataset.py`) — MediaWiki API enumeration over 36
   Mamluk/Islamic Cairo categories, filtering on a 800px minimum edge and JPEG/PNG only.
2. **Review** — every candidate was inspected visually on a contact sheet at two
   zoom levels. 123 candidates were reviewed; 39 were kept.
3. **Rejection** — 20 candidates were rejected for a recorded reason. The
   dominant failure modes were:
   - **Wrong country** — Portuguese colonial facades, Agra and Delhi monuments, Aleppo,
     the Armenian Quarter, all reaching these categories through shared vocabulary
     ("mashrabiya", "muqarnas").
   - **Not a photograph** — 19th-century lithographs (David Roberts), orientalist
     paintings (J. F. Lewis), book scans, architectural plans.
   - **Modern architecture** — 20th/21st-century apartment towers, modern mosques with
     no Mamluk vocabulary, construction cranes, night shots dominated by LED lighting.
   - **Museum objects** — mashrabiya screens photographed as artefacts, not in situ.
   Each rejection and its reason is recorded in `data/exclusions.csv`.
4. **Quality control** (`scripts/build_dataset.py`) — corrupted-file drop, perceptual-hash
   deduplication (64-bit pHash, Hamming distance <= 10), 512px centre-crop and resize,
   deterministic 85/15 split seeded at 0.

Deduplication found **0 duplicates** at that threshold; this was verified rather than
assumed, by confirming the hash function separates a JPEG-recompressed copy of an
image (distance 0) from a genuinely different image (distance 36).

## Structure

```
data/train/         33 images, 512x512 PNG
data/validation/    6 images, 512x512 PNG
metadata.csv        image_path, caption, source, architectural_style, split,
                    license, license_url, artist, title, category, phash
captions.jsonl      caption, style token, VLM base description, provenance
data/selected.csv   the curated allowlist (pageid + note)
data/exclusions.csv rejected candidates with reasons
```

## Captions

Each caption follows `[STYLE TOKEN] + description + structural elements + materials + viewpoint`,
for example:

> `mamluk_architecture, historic Islamic courtyard in Cairo, carved limestone facade, pointed arches, geometric stone ornamentation, warm natural light, architectural photography`

The architectural vocabulary is grounded in the image's recorded Wikimedia category
rather than generated freely, so the template cannot invent a minaret on a doorway.

## Limitations

- **Small** (39 images) and heavily weighted toward Cairo. Not a general Islamic
  architecture dataset.
- **Wikimedia Commons bias** — photographs are skewed toward well-documented,
  tourist-accessible monuments, and toward the aesthetics of the contributing
  photographers.
- **Not deduplicated against the wider web** — only against itself.
- Some images contain minor modern intrusions (street furniture, signage, vehicles)
  that were judged not to dominate the frame.

## Per-image attribution

| # | File | Source | Licence | Author |
|---|---|---|---|---|
| 1 | Ayyubid Wall Al-Azhar Park Cairo 01-2006.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Ayyubid_Wall_Al-Azhar_Park_Cairo_01-2006.jpg) | CC BY 3.0 | Błażej Pindor |
| 2 | Islamic Cairo63.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Islamic_Cairo63.jpg) | CC BY-SA 4.0 | Mohamed Ouda |
| 3 | A small mosque in the center of Cairo, Egypt, North Africa.jpg | [Commons](https://commons.wikimedia.org/wiki/File:A_small_mosque_in_the_center_of_Cairo,_Egypt,_North_Africa.jpg) | CC BY-SA 3.0 | Mstyslav Chernov |
| 4 | Qaytbay sabil exterior decoration2.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Qaytbay_sabil_exterior_decoration2.jpg) | CC BY-SA 4.0 | Robert Prazeres |
| 5 | The Mausoleum of Sultan Qalawun.jpg | [Commons](https://commons.wikimedia.org/wiki/File:The_Mausoleum_of_Sultan_Qalawun.jpg) | CC BY-SA 3.0 | Bassem Abdelaziz |
| 6 | Islamic Cairo60.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Islamic_Cairo60.jpg) | CC BY-SA 4.0 | Mohamed Ouda |
| 7 | Sultan Qalawun Mosque at Mu'izz street.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Sultan_Qalawun_Mosque_at_Mu%27izz_street.jpg) | CC BY-SA 3.0 | Mariam Mohamed Kamal |
| 8 | بسطام.jpg | [Commons](https://commons.wikimedia.org/wiki/File:%D8%A8%D8%B3%D8%B7%D8%A7%D9%85.jpg) | CC BY-SA 4.0 | Peymanpourkoushki |
| 9 | Cairo (4015028718).jpg | [Commons](https://commons.wikimedia.org/wiki/File:Cairo_(4015028718).jpg) | CC BY-SA 2.0 | M M from Switzerland |
| 10 | Al-Azhar 2006.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Al-Azhar_2006.jpg) | Public domain | Tentoila |
| 11 | Cairo. Mameluke House.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Cairo._Mameluke_House.jpg) | No restrictions | Cornell University Library |
| 12 | Hard work in el darb el asfar (old cairo).jpg | [Commons](https://commons.wikimedia.org/wiki/File:Hard_work_in_el_darb_el_asfar_(old_cairo).jpg) | CC BY-SA 4.0 | Eslamashrafrefaat |
| 13 | Mosque of Gawhar al-Lala DSCF3441.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Mosque_of_Gawhar_al-Lala_DSCF3441.jpg) | CC BY-SA 4.0 | R Prazeres |
| 14 | Cairo - Islamic district - Cemetery.JPG | [Commons](https://commons.wikimedia.org/wiki/File:Cairo_-_Islamic_district_-_Cemetery.JPG) | CC BY-SA 4.0 | Daniel Mayer |
| 15 | Bayn al-Qasrayn2.JPG | [Commons](https://commons.wikimedia.org/wiki/File:Bayn_al-Qasrayn2.JPG) | CC BY-SA 3.0 | Casual Builder |
| 16 | Flickr - HuTect ShOts - Dome of Madrasa and Masjid of Sultan Qaytbay مدرسة ومسجد السلطان قايتباي - Cairo - Egypt - 28 05 2010.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Flickr_-_HuTect_ShOts_-_Dome_of_Madrasa_and_Masjid_of_Sultan_Qaytbay_%D9%85%D8%AF%D8%B1%D8%B3%D8%A9_%D9%88%D9%85%D8%B3%D8%AC%D8%AF_%D8%A7%D9%84%D8%B3%D9%84%D8%B7%D8%A7%D9%86_%D9%82%D8%A7%D9%8A%D8%AA%D8%A8%D8%A7%D9%8A_-_Cairo_-_Egypt_-_28_05_2010.jpg) | CC BY-SA 2.0 | Ahmed Al.Badawy from Cairo, Egypt |
| 17 | Museo gayer anderson, cortile 03.JPG | [Commons](https://commons.wikimedia.org/wiki/File:Museo_gayer_anderson,_cortile_03.JPG) | CC BY 3.0 | Sailko |
| 18 | Cairo2761.JPG | [Commons](https://commons.wikimedia.org/wiki/File:Cairo2761.JPG) | CC BY-SA 3.0 | Samir I. Sharbaty |
| 19 | Al-Nasir Muhammad Mosque BW 1.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Al-Nasir_Muhammad_Mosque_BW_1.jpg) | CC BY 3.0 | Berthold Werner |
| 20 | احدى غرف سكن المماليك بالحطابة.JPG | [Commons](https://commons.wikimedia.org/wiki/File:%D8%A7%D8%AD%D8%AF%D9%89_%D8%BA%D8%B1%D9%81_%D8%B3%D9%83%D9%86_%D8%A7%D9%84%D9%85%D9%85%D8%A7%D9%84%D9%8A%D9%83_%D8%A8%D8%A7%D9%84%D8%AD%D8%B7%D8%A7%D8%A8%D8%A9.JPG) | CC BY-SA 3.0 | Hanyfawwaz |
| 21 | Qaytbay sabil-kuttab.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Qaytbay_sabil-kuttab.jpg) | CC BY-SA 4.0 | Robert Prazeres |
| 22 | An alley in Muizz street..JPG | [Commons](https://commons.wikimedia.org/wiki/File:An_alley_in_Muizz_street..JPG) | CC BY-SA 3.0 | SalmaMSan |
| 23 | Sultan Hassan Mosque, Cairo.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Sultan_Hassan_Mosque,_Cairo.jpg) | CC BY-SA 4.0 | Bekriah Mawasi |
| 24 | City of the Dead (south).jpg | [Commons](https://commons.wikimedia.org/wiki/File:City_of_the_Dead_(south).jpg) | CC BY-SA 4.0 | Casual Builder |
| 25 | Complex of Al Sultan Al Zahir Barquq 011.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Complex_of_Al_Sultan_Al_Zahir_Barquq_011.jpg) | CC BY-SA 4.0 | Keladawy |
| 26 | Flickr - HuTect ShOts - Masjid Qanibay Al Ramah مسجد قاني باي الرماح - Cairo - Egypt - 17 04 2010.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Flickr_-_HuTect_ShOts_-_Masjid_Qanibay_Al_Ramah_%D9%85%D8%B3%D8%AC%D8%AF_%D9%82%D8%A7%D9%86%D9%8A_%D8%A8%D8%A7%D9%8A_%D8%A7%D9%84%D8%B1%D9%85%D8%A7%D8%AD_-_Cairo_-_Egypt_-_17_04_2010.jpg) | CC BY-SA 2.0 | Ahmed Al.Badawy from Cairo, Egypt |
| 27 | Tomb of az-Zahir Qansuh.JPG | [Commons](https://commons.wikimedia.org/wiki/File:Tomb_of_az-Zahir_Qansuh.JPG) | CC BY-SA 3.0 | Tekisch |
| 28 | Cairo, the Mosque of Sultan Hassan and the Ar-Rif (6201079985).jpg | [Commons](https://commons.wikimedia.org/wiki/File:Cairo,_the_Mosque_of_Sultan_Hassan_and_the_Ar-Rif_(6201079985).jpg) | CC BY 2.0 | Arian Zwegers from Brussels, Belgium |
| 29 | A small mosque in the center of Cairo, Egypt, North Africa-3.jpg | [Commons](https://commons.wikimedia.org/wiki/File:A_small_mosque_in_the_center_of_Cairo,_Egypt,_North_Africa-3.jpg) | CC BY-SA 3.0 | Mstyslav Chernov |
| 30 | الزخارف العباسية.jpg | [Commons](https://commons.wikimedia.org/wiki/File:%D8%A7%D9%84%D8%B2%D8%AE%D8%A7%D8%B1%D9%81_%D8%A7%D9%84%D8%B9%D8%A8%D8%A7%D8%B3%D9%8A%D8%A9.jpg) | CC BY-SA 4.0 | Mohammed Harith Khalil |
| 31 | Cairo, madrasa del sultano qalaun, 01.JPG | [Commons](https://commons.wikimedia.org/wiki/File:Cairo,_madrasa_del_sultano_qalaun,_01.JPG) | CC BY 3.0 | Sailko |
| 32 | Cairouni.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Cairouni.jpg) | CC BY-SA 3.0 | user:mmustafa |
| 33 | Door in Al-Gamaliya street, photo by Hatem Moushir 01.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Door_in_Al-Gamaliya_street,_photo_by_Hatem_Moushir_01.jpg) | CC BY-SA 4.0 | Hatem Moushir |
| 34 | Une vue en hauteur du caire.JPG | [Commons](https://commons.wikimedia.org/wiki/File:Une_vue_en_hauteur_du_caire.JPG) | CC BY-SA 3.0 | Papillus |
| 35 | Flickr - Gaspa - Cairo, una casa storica.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Flickr_-_Gaspa_-_Cairo,_una_casa_storica.jpg) | CC BY 2.0 | Francesco Gasparetti from Senigallia, Italy |
| 36 | Aqmar Mosque.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Aqmar_Mosque.jpg) | CC BY-SA 3.0 | Md iet ( talk ) |
| 37 | Cairo,Qarafa1.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Cairo,Qarafa1.jpg) | CC BY-SA 3.0 | Bertramz |
| 38 | QalaunMosque.JPG | [Commons](https://commons.wikimedia.org/wiki/File:QalaunMosque.JPG) | CC BY-SA 3.0 | المصري الأصيل |
| 39 | Door in Al-Gamaliya street, photo by Hatem Moushir 02.jpg | [Commons](https://commons.wikimedia.org/wiki/File:Door_in_Al-Gamaliya_street,_photo_by_Hatem_Moushir_02.jpg) | CC BY-SA 4.0 | Hatem Moushir |
