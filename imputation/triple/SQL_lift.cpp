

#include <triple/SQL_lift.h>

#include <memory>
#include <stdexcept>
template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}


std::string Triple::lift(std::vector<std::string> attributes){

    std::string query = "struct_pack(N := 1, lin_num :=%s, quadratic_num := %s)";
    std::string lin_agg = "LIST_VALUE(";
    std::string quad_agg = "LIST_VALUE(";


    for (size_t i=0;i<attributes.size();i++){
        lin_agg += (attributes[i]+",");
        for(size_t j=i;j<attributes.size();j++)
            quad_agg += attributes[i]+"*"+attributes[j]+",";//a*a, a*b, a*c, b*b, b*c, c*c
    }

    lin_agg.pop_back();
    lin_agg+=")";

    quad_agg.pop_back();
    quad_agg+=")";

    return string_format(query, lin_agg.c_str(), quad_agg.c_str());
}
