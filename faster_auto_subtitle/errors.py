class SubtitleError(Exception):
    """Error with a message meant for the user, reported without a traceback."""


class ModelNotCachedError(SubtitleError):
    """A model or other resource is missing from the local cache while offline."""


class TranslationUnavailableError(SubtitleError):
    """No translation path exists between the requested languages."""
