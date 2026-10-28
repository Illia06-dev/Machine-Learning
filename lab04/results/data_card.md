# Data Card — UCI Adult

## Source
UCI Machine Learning Repository, Adult dataset (Becker & Kohavi, 1996), extracted from the 1994 US Census.
Loaded via `fetch_openml(name='adult', version=2, as_frame=True)`.

## License
CC BY 4.0 (UCI Machine Learning Repository).

## Size
- Raw: 48842 rows, 15 columns (14 features + income).
- Processed train: 34165 rows, 108 columns.
- Processed test: 14653 rows, 108 columns.
- Processed SHA256: `b72746551ad1a1c82271227f7a805b3f3a68878ad46c30b15acafdcfad68eb82`

## Known biases
- Data from 1994; the $50K threshold is not adjusted for inflation.
- Class imbalance: 23.9% of rows are `>50K`.
- Sex imbalance: Male 66.8%, Female 33.2%.
- Race imbalance: White 85.5%, Black 9.6%, Asian-Pac-Islander 3.1%, Amer-Indian-Eskimo 1.0%, Other 0.8%.
- 89.7% of rows are from the United States; other countries are under-represented.
- Missing values ('?') are concentrated in workclass, occupation and native-country.

## Privacy concerns
- Census microdata about real people; contains sensitive attributes (race, sex, marital status, country of origin, income).
- No names or IDs, but combinations of age, sex, race, country and occupation can re-identify rare individuals.
