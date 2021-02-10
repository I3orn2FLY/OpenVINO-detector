//
// Created by dmitry.fedorov on 4.11.19.
//

#pragma once
typedef long long Timespan;

class Timestamp {
    unsigned long long _time;

public:
    Timestamp() : _time(0) {}

    explicit Timestamp(unsigned long long t) : _time(t) {}

    Timestamp &operator=(unsigned long long t) {
        _time = t;
        return *this;
    }

    bool operator>=(const Timestamp &t) const {
        return (_time - t._time) < 0x7fffffff;
    }

    bool operator<=(const Timestamp &t) const {
        return (t >= *this);
    }

    bool operator<(const Timestamp &t) const {
        return !operator>=(t);
    }

    bool operator>(const Timestamp &t) const {
        return !operator<=(t);
    }

    bool operator==(const Timestamp &t) const {
        return _time == t._time;
    }

    bool operator!=(const Timestamp &t) const {
        return _time != t._time;
    }

    Timestamp operator+(const Timespan &d) const {
        return Timestamp(_time + d);
    }

    Timestamp operator-(const Timespan &d) const {
        return Timestamp(_time - d);
    }

    Timespan operator-(const Timestamp &t) const {
        return _time - t._time;
    }

    bool operator!() const {
        return _time == 0;
    }

    operator bool() const {
        return _time != 0;
    }

    operator unsigned long long() const {
        return _time;
    }

    operator long long() const {
        return (long long) _time;
    }

    operator double() const {
        return (double) _time;
    }

    inline double ToSecs() const {
        const double MS_IN_SEC = 1000.0;
        return _time / MS_IN_SEC;
    }

    inline static Timestamp FromSecs(double time) {
        assert(time >= 0);
        assert(time <= std::numeric_limits<unsigned long long>::max());
        const double MS_IN_SEC = 1000.0;
        return Timestamp((unsigned long long) (time * MS_IN_SEC));
    }

    friend inline std::ostream &operator<<(std::ostream &os, const Timestamp &t);

    friend inline std::istream &operator>>(std::istream &is, Timestamp &t);

    static Timestamp Now() {
        return Timestamp(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }

    static Timestamp NowUSec() {
        return Timestamp(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }

    Timespan GetYear() {
        std::time_t timeT = _time / 1000; // to seconds
        struct tm *tnow = std::gmtime(&timeT);
        return tnow->tm_year + 1900;
    }

    Timespan GetMon() {
        std::time_t timeT = _time / 1000; // to seconds
        struct tm *tnow = std::gmtime(&timeT);
        return tnow->tm_mon + 1;
    }

    Timespan GetDay() {
        std::time_t timeT = _time / 1000; // to seconds
        struct tm *tnow = std::gmtime(&timeT);
        return tnow->tm_mday;
    }

    Timespan GetHour() {
        std::time_t timeT = _time / 1000; // to seconds
        struct tm *tnow = std::gmtime(&timeT);
        return tnow->tm_hour;
    }

    Timespan GetMin() {
        std::time_t timeT = _time / 1000; // to seconds
        struct tm *tnow = std::gmtime(&timeT);
        return tnow->tm_min;
    }

    std::string ToDateTimeString(std::string fmt = "%d %m %Y %H:%M:%S") {
        std::time_t timeT = _time / 1000; // to seconds
        std::tm timeTM = *std::localtime(&timeT);
        std::stringstream ss;
        ss << std::put_time(&timeTM, fmt.c_str());
        return ss.str();
    }

    std::string ToUnicodeTSString() {
        auto ret = std::to_string(_time / 1000); // convert to seconds
        return std::string(ret.size() < 12 ? 12 - ret.size() : 0, '0').append(
                ret); // 12 digits enough to count next century
    }
};

inline std::ostream &operator<<(std::ostream &os, const Timestamp &t) {
    os << t._time;
    return os;
}

inline std::istream &operator>>(std::istream &is, Timestamp &t) {
    is >> t._time;
    return is;
}


#define ST_GET_TIMESTAMP()          Timestamp::Now()
#define ST_GET_TIMESTAMP_USEC()     Timestamp::NowUSec()