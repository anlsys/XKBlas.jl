module Logger

    import ..XKBlas

    function info(s)
        XKBlas.xkrt_logger_info(s)
    end

    function debug(s)
        XKBlas.xkrt_logger_debug(s)
    end

    function warn(s)
        XKBlas.xkrt_logger_warn(s)
    end

    function error(s)
        XKBlas.xkrt_logger_error(s)
    end

    function fatal(s)
        XKBlas.xkrt_logger_fatal(s)
    end

end
