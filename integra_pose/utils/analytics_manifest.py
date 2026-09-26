"""Version contract shared by analytics output and its consumers."""

CURRENT_SCHEMA_VERSION = 4
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3, 4)


def validate_schema_version(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Manifest JSON must be an object.")
    version = manifest.get("schema_version")
    if type(version) is not int or version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(map(str, SUPPORTED_SCHEMA_VERSIONS))
        raise ValueError(
            f"Unsupported or missing schema_version (expected one of {supported}, got {version!r})."
        )
    return version
