from datetime import datetime, timedelta

def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}

    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(
            f"Expected one of {set(str2val.keys())}, got {string}")

def str2timeinterval(string):
    if string is None:
        return None
    
    if '-' not in string:
        raise ValueError(
            f"Expected time interval HH:mm:ss-HH:mm:ss or HH:mm-HH:mm or ss-ss, got {string}")
    
    intervals = string.split('-')
    if len(intervals) != 2:
        raise ValueError(
            f"Expected time interval HH:mm:ss-HH:mm:ss or HH:mm-HH:mm or ss-ss, got {string}")

    start = try_parse_timestamp(intervals[0])
    end = try_parse_timestamp(intervals[1])
    if start >= end:
        raise ValueError(
            f"Expected time interval end to be higher than start, got {start} >= {end}")
    
    return [start, end]

def time_to_timestamp(string):
    split_time = string.split(':')
    if len(split_time) == 0 or len(split_time) > 3 or not all([ x.isdigit() for x in split_time ]):
        raise ValueError(
            f"Expected HH:mm:ss or HH:mm or ss, got {string}")
    
    if len(split_time) == 1:
        return int(split_time[0])
    
    if len(split_time) == 2:
        return int(split_time[0]) * 60 * 60 + int(split_time[1]) * 60
    
    return int(split_time[0]) * 60 * 60 + int(split_time[1]) * 60 + int(split_time[2])

def try_parse_timestamp(string):
    timestamp = parse_timestamp(string, '%H:%M:%S')
    if timestamp is not None:
        return timestamp
    
    timestamp = parse_timestamp(string, '%H:%M')
    if timestamp is not None:
        return timestamp
    
    return parse_timestamp(string, '%S')

def parse_timestamp(string, pattern):
    try:
        date = datetime.strptime(string, pattern)
        delta = timedelta(hours=date.hour, minutes=date.minute, seconds=date.second)
        return int(delta.total_seconds())
    except:
        return None

def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"

