module Logger

    import ..XK

    function info(s)
        XK.xkrt_logger_info(s)
    end

    function debug(s)
        XK.xkrt_logger_debug(s)
    end

    function warn(s)
        XK.xkrt_logger_warn(s)
    end

    function error(s)
        XK.xkrt_logger_error(s)
    end

    function fatal(s)
        XK.xkrt_logger_fatal(s)
    end

end
