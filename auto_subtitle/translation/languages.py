import langcodes
from transformers.models.marian.convert_marian_tatoeba_to_pytorch import GROUP_MEMBERS


def to_alpha2_languages(languages: list[str]) -> set[str]:
    return set(item for sublist in [__to_alpha2_language(language) for language in languages] for item in sublist)


def __to_alpha2_language(language: str) -> set[str]:
    if len(language) == 2:
        return {language}

    if language in GROUP_MEMBERS:
        return set([langcodes.Language.get(x).language for x in GROUP_MEMBERS[language][1]])

    return {langcodes.Language.get(language).language}


def to_alpha3_language(language: str) -> str:
    return langcodes.Language.get(language).to_alpha3()
