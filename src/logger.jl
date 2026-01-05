module Logger

    import ..XKLas

    function info(s)
        XKLas.xkrt_logger_info(s)
    end

    function debug(s)
        XKLas.xkrt_logger_debug(s)
    end

    function warn(s)
        XKLas.xkrt_logger_warn(s)
    end

    function error(s)
        XKLas.xkrt_logger_error(s)
    end

    function fatal(s)
        XKLas.xkrt_logger_fatal(s)
    end

end
