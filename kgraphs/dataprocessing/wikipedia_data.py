from ..utils.logging import MAIN_LOGGER_NAME, create_logger


def articlestr_to_wellformatted(dic: dict):
    logger = create_logger(MAIN_LOGGER_NAME)
    assertions = ["sections" in dic["parse"], "wikitext" in dic["parse"]]  # type: ignore
    assert all(
        assertions
    ), "Invalid dictionary structure passed to articlestr_to_wellformatted()"
    # Get all section offsets
    sections = dic["parse"]["sections"]  # type: ignore
    whole_text = dic["parse"]["wikitext"]["*"]  # type:ignore
    # Get all sections by byte offset

    for bo_idx in range(len(sections) - 1):
        section = sections[bo_idx]
        level = section["level"]  # TODO: Might be useful later
        start_offset = section["byteoffset"]  # type: ignore
        end_offset = sections[bo_idx + 1]["byteoffset"]  # type:ignore

        section_name = section["line"]  # type:ignore
        section_text = whole_text[start_offset:end_offset]
        logger.debug(f"Section '{section_name}':\n{section_text}")
