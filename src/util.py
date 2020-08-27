#!/usr/bin/env python3


def get_span_indexes(sentence, span):
    """
    Method which returns the indexes of a substring of a string, if such substring
    exists.
    :param sentence: The sentence string.
    :param span:     A substring to be searched for.
    :return:         (start_index, end_index) tuple.
    """
    try:
        start_index = str(sentence).index(span)
        end_index = start_index + len(span)
        return start_index, end_index
    except ValueError:
        return (-1, -1)


def get_spans_indexes(sentence, spans):
    """
    Same as get_span_indexes function, but takes a list of spans instead of a single span.
    :param sentence: The sentence string.
    :param spans:    List of substrings to be searched for.
    :return:         List of (start_index, end_index) pairs.
    """
    RESULT = []
    for span in spans:
        res = get_span_indexes(sentence, span)
        RESULT.append(res)
    return RESULT


def print_matches(match_text, entities_dict):
    ent_list = entities_dict['entities'] # list of entities in the form of (start_index, end_index, label)
    for (start, end, label) in ent_list:
        print(match_text[start:end], label)


if __name__ == '__main__':
    phrase = (
        "I landed without clearance. We landed on Approach Control frequency. While clearing runway I realized we were"
        " still on Approach Control frequency. I heard transmissions that sounded like what an Approach Controller "
        "would say - not Tower. I asked my partner 'were we cleared to land?' My partner quickly switched to Tower and"
        " we heard him clearing us to cross parallel runway and contact Ground. we acknowledged and complied. Tower "
        "Controller then said he tried to call us a couple of times. My partner told him that Approach Control never "
        "switched us over. The last thing I remember on approach was the Approach Controller issuing us traffic over "
        "the stadium; that was it. I do not recall ever being told to contact the Tower. We were also at the end of "
        "two very long days dealing with multiple issues and worn down. in this mental state I was not sharp enough "
        "to switch to Tower on my own past the marker without being prompted to do so by ATC. It was quiet and smooth."
        " So many approaches and landings in the last four days - it is hard to keep track because they start running"
        " together being so automatic. We were fatigued towards the end of this flight. After listening to LIVEATC.com"
        " it sounds like we were forgotten about. The Tower asked us where we were parking close to a short final; "
        "queried us again; and then again and issued the taxi instructions. Which is when we switched over and heard "
        "them. It does not seem as though they were looking for us long. I take full responsibility for landing "
        "without a clearance. I always wondered how you could land without a clearance and now I know. Contributing "
        "factors were crew fatigue and lack of ATC Comm. Normally I would query ATC about switching to Tower; or just "
        "switch myself past the marker if they are busy; but due to fatigue I did not catch it and we continued. To "
        "help prevent this in the future I will monitor my fatigue level better.")

    print(get_spans_indexes(phrase, ['transmissions', 'dkfslaf', 'fatigue']))


